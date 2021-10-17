import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from graph_conv import AugGraphConv


def dropout_mask(x, sz, p: float):
    "Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element."
    return x.new(*sz).bernoulli_(1-p).div_(1-p)


class CrossEmbedding(nn.Module):

    initrange = 0.1

    def __init__(self, n_hid, pretrain=None, n_emb=None, padding_idx=None, always_freeze=True):
        super().__init__()
        assert pretrain is not None or n_emb is not None
        if pretrain is not None:
            if always_freeze:
                self.register_buffer('emb', pretrain.float())
            else:
                self.emb = nn.Embedding.from_pretrained(pretrain.float())
            self.num_embeddings = pretrain.size(0)
            self.transform = nn.Linear(pretrain.size(1), n_hid, bias=False)
        else:
            self.num_embeddings = n_emb
            self.emb = nn.Embedding(n_emb, n_hid, padding_idx=padding_idx)
            self.emb.weight.data.uniform_(-self.initrange, self.initrange)
            if self.emb.padding_idx is not None:
                self.emb.weight.data[self.emb.padding_idx] = 0.
            self.transform = None

        self.padding_idx = padding_idx

    def forward(self, word_idx=None):
        if isinstance(self.emb, nn.Embedding):
            emb = self.emb.weight
        else:
            emb = self.emb.detach().clone()

        if word_idx is not None:
            emb = F.embedding(word_idx, emb, padding_idx=self.padding_idx)
        if self.transform is not None:
            emb = self.transform(emb)

        return emb

    def extra_repr(self):
        return 'num_embeddings={}'.format(self.num_embeddings)


def resize_heads(n_hid, n_heads):
    for head in (10, 12, 16):
        if n_hid % head == 0:
            return n_heads


def updated_hook(m, i, o):
    for _m in m.modules():
        if hasattr(_m, '_need_update'):
            _m._need_update = True


class DHGNet(nn.Module):

    def train(self, mode: bool = True):
        self._need_update = True
        return super().train(mode=mode)

    def __init__(self, n_hid, edge_index, edge_type, node_type, offset, pretrains, conv_name='hgnn', n_layers=2, dropout=0.05, n_heads=10, layer_norm=True, always_freeze=True, padding_idx=None, neg=10, temp=0.1, always_update=False, residual=True, align_batch_size=50000):
        super().__init__()
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_type', edge_type)
        self.register_buffer('node_type', node_type)
        self.register_buffer('offset', offset)

        if n_hid % n_heads != 0:
            resize_heads(n_hid, n_heads)
            print(f'changing number of GNN heads to {n_heads}')

        self.n_hid = n_hid
        self.padding_idx = padding_idx
        self.num_embeddings = offset[1] - offset[0]
        self.weight = torch.zeros([self.num_embeddings, n_hid])

        self.word_embs = nn.ModuleList()
        for i, pretrain in enumerate(pretrains):
            n_emb = offset[i+1] - offset[i]
            if i == 0:
                self.word_embs.append(CrossEmbedding(
                    n_hid, pretrain, n_emb, padding_idx=padding_idx, always_freeze=always_freeze))
            else:
                self.word_embs.append(CrossEmbedding(
                    n_hid, pretrain, n_emb, padding_idx=None, always_freeze=always_freeze))

        num_types = len(pretrains)
        num_relations = edge_type.max().item() + 1
        self.drop = nn.Dropout(dropout)

        self.graph_convs = nn.ModuleList()
        if n_layers > 0:
            for l in range(n_layers - 1):
                self.graph_convs.append(AugGraphConv(conv_name, n_hid, n_hid, num_types,
                                        num_relations, n_heads, dropout, layer_norm=layer_norm, residual=residual))
            self.graph_convs.append(AugGraphConv(conv_name, n_hid, n_hid, num_types, num_relations,
                                    n_heads, dropout, nonlinear=False, layer_norm=layer_norm, residual=residual))

        self._need_update = True
        # self.register_backward_hook(self._updated_hook)

        self._similarity = nn.CosineSimilarity(dim=-1)
        self.neg = neg
        self.temp = temp
        self.always_update = always_update
        self.align_batch_size = align_batch_size

    def forward(self, word_idx):
        weight = self.get_embedding()
        return F.embedding(word_idx, weight, self.padding_idx)

    def aligned_loss(self, features=None, batch_size=50000):
        if features is None:
            features = self.get_initial_node_features()
        return self._graph_loss(features, batch_size=batch_size)

    def _node_similarity(self, node_rep, node_head, node_tail):
        return self._similarity(node_rep[node_head], node_rep[node_tail])

    def _graph_loss(self, node_rep, batch_size):
        if self.edge_index.size(1) > batch_size:
            idx = torch.randperm(self.edge_index.size(
                1), device=self.edge_index.device)[:batch_size]
            head, tail = self.edge_index[0][idx], self.edge_index[1][idx]
        else:
            head, tail = self.edge_index[0], self.edge_index[1]
        pos_sim = self._node_similarity(node_rep, head, tail).unsqueeze(-1)
        neg_sim_head = self._node_similarity(
            node_rep, self._negative_sample(head, self.neg), tail.unsqueeze(-1))
        neg_sim_tail = self._node_similarity(
            node_rep, head.unsqueeze(-1), self._negative_sample(tail, self.neg))
        logits = torch.cat([pos_sim, neg_sim_head, neg_sim_tail], dim=-1)
        logits /= self.temp

        labels = torch.zeros_like(head)
        loss = F.cross_entropy(logits, labels)
        return loss

    def _negative_sample(self, positives, n_repeat=1):
        if not hasattr(self, 'node_pools'):
            nodes = torch.unique(self.edge_index)
            self.node_pools = {}
            for i in range(len(self.word_embs)):
                self.node_pools[i] = nodes[self.node_type[nodes] == i]
        node_type = self.node_type[positives]
        negatives = positives.new_zeros([*positives.shape, n_repeat])
        for i in range(len(self.word_embs)):
            mask = node_type == i
            n_mask = mask.sum().item()
            if n_mask == 0:
                continue
            rand_idx = torch.randint(len(self.node_pools[i]), size=[
                                     n_mask, n_repeat], dtype=positives.dtype, device=positives.device)
            negatives[mask] = self.node_pools[i][rand_idx]
        return negatives

    def get_initial_node_features(self):
        features = []
        for i, word_emb in enumerate(self.word_embs):
            feature = word_emb()
            features.append(feature)
        features = torch.cat(features, dim=0)
        features = self.drop(features)
        return features

    def get_embedding(self, keep_attention=False):
        if not self._need_update:
            return self.weight
        features = self.get_initial_node_features()
        node_type, edge_index, edge_type = self.node_type.clone(
        ), self.edge_index.clone(), self.edge_type.clone()
        for i, gc in enumerate(self.graph_convs):
            if i == len(self.graph_convs) - 1:
                mask = edge_index[1] < self.num_embeddings
                edge_index = edge_index[:, mask]
                edge_type = edge_type[mask]
            features = gc(features, node_type, edge_index,
                          edge_type, keep_attention=keep_attention)

        self.weight = features[:self.num_embeddings]
        self.weight[self.padding_idx] = 0
        self._need_update = self.always_update
        return self.weight


class EmbeddingDropout(nn.Module):
    "Apply dropout with probabily `embed_p` to an embedding layer `emb`."

    def __init__(self, gnn_emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = gnn_emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1
        # self.register_backward_hook(self._updated_hook)

    def forward(self, words, scale=None):
        weight = self.emb.get_embedding()
        if self.training and self.embed_p != 0:
            size = (weight.size(0), 1)
            mask = dropout_mask(weight.data, size, self.embed_p)
            masked_embed = weight * mask
        else:
            masked_embed = weight
        if scale:
            masked_embed = masked_embed * scale
        return F.embedding(words, masked_embed, self.pad_idx)

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def extra_repr(self):
        return 'embed_p={}'.format(self.embed_p)


class TiedGNNLinear(nn.Module):

    def __init__(self, gnn_emb, bias=True):
        super().__init__()
        self.gnn_emb = gnn_emb
        device = list(gnn_emb.parameters())[0].device
        self.out_features, self.in_features = gnn_emb.weight.shape
        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                gnn_emb.weight.size(0))).to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.gnn_emb.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, word_idx):
        return F.linear(word_idx, self.gnn_emb.weight, self.bias)

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
