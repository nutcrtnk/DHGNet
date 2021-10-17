import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import GELU
import torch_geometric.nn as G
from torch_geometric.nn import conv
from torch_geometric.utils import k_hop_subgraph


class Bypass(nn.Module):

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        return args


class HGNNConv(nn.Module):

    def __init__(self, in_dim, out_dim, num_relations, heads=8, word_conv='gat', **kwargs):
        super().__init__()

        self.num_relations = num_relations
        self.self_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.act = GELU()
        self.word_conv = nn.ModuleList()
        
        for _ in range(self.num_relations):
            if word_conv == 'gat':
                self.word_conv.append(G.GATConv(in_channels=in_dim, out_channels=out_dim // heads, heads=heads,
                                        concat=True, add_self_loops=False, **kwargs))
            elif word_conv == 'gcn':
                self.word_conv.append(G.GCNConv(in_channels=in_dim, out_channels=out_dim, add_self_loops=False))
            else:
                raise Exception('invalid word-level aggregation options')
        self.lang_conv = G.GATConv(
            [out_dim, in_dim], out_dim // heads, heads=heads, add_self_loops=False, concat=True)
        self.lang_conv.lin_l = Bypass()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_cross = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, node_inp, edge_index, edge_type, keep_attention=False):
        rel_target_nodes = []
        rel_edge_type = []
        rel_inputs = []
        device = edge_index.device

        rel_inputs.append(self.lin_self(node_inp))
        rel_edge_type.append(edge_index.new_ones(node_inp.size(0)) * -1)
        rel_target_nodes.append(torch.arange(node_inp.size(0), device=device))

        if keep_attention:
            self.att = {}
        for rel in range(self.num_relations):
            mask = edge_type.flatten() == rel
            sub_edge_index = edge_index[:, mask]
            if sub_edge_index.size(1) == 0:
                continue
            target_nodes = torch.unique(sub_edge_index[1, :])
            rel_target_nodes.append(target_nodes)
            sub_nodes, sub_edge_index, mapping, mask2 = k_hop_subgraph(
                target_nodes, 1, sub_edge_index, relabel_nodes=True)
            if keep_attention and isinstance(self.word_conv[rel], G.GATConv):
                out, (_, _att) = self.word_conv[rel](
                    node_inp[sub_nodes], sub_edge_index, return_attention_weights=True)
                self.att[rel] = (edge_index[:, mask][:, mask2], _att.detach())
            else:
                out = self.word_conv[rel](node_inp[sub_nodes], sub_edge_index)
            out = self.act(out[mapping])
            out = self.lin_cross(out)
            rel_inputs.append(out)
            rel_edge_type.append(edge_index.new_ones(out.size(0)) * rel)
        rel_inputs = torch.cat(rel_inputs, dim=0)
        rel_target_nodes = torch.cat(rel_target_nodes)
        rel_edge_index = torch.stack(
            [torch.arange(len(rel_target_nodes), device=device), rel_target_nodes])
        rel_edge_type = torch.cat(rel_edge_type, dim=0)
        out, (rel_edge_index, att) = self.lang_conv(
            rel_inputs, rel_edge_index, return_attention_weights=True)
        out = out[:len(node_inp)]
        if keep_attention:
            self.att[-1] = (rel_edge_index, att.detach(), rel_edge_type)
        return out


class AugGraphConv(nn.Module):
    def __init__(self, conv_name, in_channels, out_channels, num_types, num_relations, heads, dropout, nonlinear=True, residual=True, layer_norm=False):
        super().__init__()
        self.conv_name = conv_name
        if conv_name == 'gcn':
            self.conv = G.GCNConv(in_channels, out_channels)
        elif conv_name == 'gat':
            self.conv = G.GATConv(
                in_channels, out_channels // heads, heads=heads)
        elif conv_name == 'hgnn':
            self.conv = HGNNConv(in_channels, out_channels,
                                 num_relations=num_relations, heads=heads)
        elif conv_name == 'hgnn-gcn':
            self.conv = HGNNConv(in_channels, out_channels,
                                 num_relations=num_relations, heads=heads, word_conv='gcn')            
        else:
            raise Exception('invalid graph conv_name')

        if nonlinear:
            self.nonlinear = nn.GELU()
        else:
            self.nonlinear = None

        self.dropout = nn.Dropout(dropout)

        self._att = None
        self._edge_index = None

        if residual:
            self.residual = lambda x, y: x + y
        else:
            self.residual = None

        if layer_norm:
            self.layer_norm = nn.LayerNorm(
                out_channels, elementwise_affine=False)
        else:
            self.layer_norm = None

    def forward(self, x_inp, node_type, edge_index, edge_type, keep_attention=False):
        x_new = None
        x = self.layer_norm(x_inp) if self.layer_norm is not None else x_inp
        if len(node_type.shape) == 1:
            node_type = node_type.unsqueeze(-1)
        if self.conv_name in {'gat', 'gcn'}:
            x_new = self.conv(x, edge_index)
        elif self.conv_name.startswith('hgnn'):
            x_new = self.conv(x, edge_index, edge_type,
                              keep_attention=keep_attention)
        assert x_new is not None
        if self.nonlinear is not None:
            x_new = self.nonlinear(x_new)
        x_new = self.dropout(x_new)
        if self.residual is not None:
            x_new = self.residual(x_new, x_inp)
        return x_new
