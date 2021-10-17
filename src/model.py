import torch
import torch.nn as nn

from fastai.text.transform import Vocab, defaults, UNK
from fastai.text.models.awd_lstm import EmbeddingDropout
from fastai.callbacks.hooks import HookCallback, Hooks

import dhgnet
import dhg

GRAPH = None


class NeedUpdateCallback(HookCallback):

    def on_train_begin(self, **kwargs):
        self.hooks = Hooks(self.modules, self.hook,
                           is_forward=False, detach=False)

    def on_epoch_begin(self, **kwargs):
        for _m in self.learn.model.modules():
            if hasattr(_m, '_need_update'):
                _m._need_update = True

    def on_train_end(self, **kwargs):
        for _m in self.learn.model.modules():
            if hasattr(_m, '_need_update'):
                _m._need_update = True
        super().on_train_end(**kwargs)

    def hook(self, m, i, o):
        for _m in m.modules():
            if hasattr(_m, '_need_update'):
                _m._need_update = True


def get_first_layer(model, module_type):
    layer_name, layer_module = None, None
    for n, m in model.named_modules():
        if isinstance(m, module_type):
            layer_name, layer_module = n, m
            break
    return layer_name, layer_module


def get_last_layer(model, module_type):
    layer_name, layer_module = None, None
    for n, m in model.named_modules():
        if isinstance(m, module_type):
            layer_name, layer_module = n, m
    return layer_name, layer_module


def set_layer(model, layer_name, new_layer):
    names = layer_name.split('.')
    layer = model
    for name in names[:-1]:
        layer = getattr(layer, name)
    setattr(layer, names[-1], new_layer)


def replace_emb_with_gnn(model, emb_layer_name, emb_layer, **gnn_config):
    device = emb_layer.weight.device
    gnn_emb = dhgnet.DHGNet(
        emb_layer.embedding_dim, padding_idx=emb_layer.padding_idx, **gnn_config).to(device)
    set_layer(model, emb_layer_name, gnn_emb)
    return gnn_emb


def create_graph(args, vocab: Vocab):
    graph = dhg.DHG(dict_name=args.dict)
    graph.add_lang(args.target_lang)
    for lang in args.langs.split(','):
        graph.add_lang(lang, w2v=args.w2v)
    vocab_size = len(vocab.itos)

    if args.add_from_dict > 0:
        dict_words = {lang: sorted(w2w.word2x.items(
        ), key=lambda x: x[1]) for lang, w2w in graph.dictionary[args.target_lang].items()}
        num_add = 0
        i = 0
        while num_add < args.add_from_dict:
            is_exceed = True
            for lang, words in dict_words.items():
                if i < len(words):
                    is_exceed = False
                    k, _ = words[i]
                    k = k.lower()
                    has_trans = False
                    w2w = graph.dictionary[args.target_lang][lang]
                    if k not in w2w.word2x:
                        continue
                    for tword in w2w(k):
                        if tword in graph.w2v_dict[lang]:
                            has_trans = True
                            break
                    if not has_trans:
                        continue
                    if k not in vocab.stoi or vocab.stoi[k] == vocab.stoi[UNK]:
                        vocab.stoi[k] = len(vocab.itos)
                        vocab.itos.append(k)
                        num_add += 1
                        if num_add >= args.add_from_dict:
                            break
            i += 1
            if is_exceed:
                break

    if args.add_from_dict > 0:
        print(f'Increase number of vocabs: {vocab_size} -> {len(vocab.itos)}')
    
    all_special_ids = set([vocab.itos.index(tok)
                          for tok in defaults.text_spec_tok])
    stoi = {k: i for k, i in vocab.stoi.items() if i not in all_special_ids}
    graph.vocabs[args.target_lang] = stoi

    graph.create_graph()
    print('finish creating dictionary graph')

    if args.save_temp:
        graph.dump_w2v_temp()
    return graph


def get_gnn_config(args, graph: dhg.DHG):
    edge_index, edge_type, node_type, offset, relation_langs = graph.get_graph_data(
        self_loop=False, reverse=args.reverse)

    print(f'number of nodes: {len(node_type)}, edges: {edge_index.size(1)}')
    print(
        f'number of found target nodes: {(node_type[torch.unique(edge_index[1])] == 0).sum().item()} / {(node_type == 0).sum().item()}')
    pretrains = graph.get_pretrain()
    gnn_config = dict(edge_index=edge_index, edge_type=edge_type, node_type=node_type, offset=offset, pretrains=pretrains,
                      conv_name=args.conv, n_layers=args.gnn_layers, residual=args.residual, layer_norm=args.layer_norm, always_freeze=args.freeze_cross)
    return gnn_config


def modify_language_model(args, learn, vocab, config, is_fastai=True):
    global GRAPH
    assert GRAPH is not None or vocab is not None

    if GRAPH is None:
        GRAPH = create_graph(args, vocab)
    gnn_config = get_gnn_config(args, GRAPH)
    gnn_config['always_update'] = not is_fastai

    replace_layers = {}

    emb_name, emb_layer = get_first_layer(learn.model, nn.Embedding)
    gnn_emb = replace_emb_with_gnn(learn.model, emb_name, emb_layer, **gnn_config)
    replace_layers[emb_layer] = gnn_emb

    emb_dp_name, emb_dp_layer = get_first_layer(learn.model, EmbeddingDropout)
    if emb_dp_name is not None:
        gnn_emb_dp = dhgnet.EmbeddingDropout(gnn_emb, config['embed_p'])
        set_layer(learn.model, emb_dp_name, gnn_emb_dp)
        replace_layers[emb_dp_layer] = gnn_emb_dp

    if config['tie_weights']:
        tied_gnn_linear = dhgnet.TiedGNNLinear(
            gnn_emb, bias=config['out_bias'])
        dec_name, dec_layer = get_last_layer(learn.model, nn.Linear)
        replace_layers[dec_layer] = tied_gnn_linear
        set_layer(learn.model, dec_name, tied_gnn_linear)

    if is_fastai:
        update_cb = NeedUpdateCallback(learn, list(replace_layers.values()))
        learn.callbacks.append(update_cb)

        for i, layer_group in enumerate(learn.layer_groups):
            if layer_group in replace_layers:
                learn.layer_groups[i] = replace_layers[layer_group]
            for name, layer in layer_group.named_modules():
                if layer in replace_layers:
                    set_layer(layer_group, name, replace_layers[layer])

    return gnn_emb


def modify_classification_model(args, learn, vocab, config, is_fastai=True):
    global GRAPH
    assert GRAPH is not None or vocab is not None

    if GRAPH is None:
        GRAPH = create_graph(args, vocab)

    gnn_config = get_gnn_config(args, GRAPH)
    gnn_config['always_update'] = not is_fastai

    replace_layers = {}
    emb_name, emb_layer = get_first_layer(learn.model, nn.Embedding)
    gnn_emb = replace_emb_with_gnn(learn.model, emb_name, emb_layer, **gnn_config)
    replace_layers[emb_layer] = gnn_emb

    emb_dp_name, emb_dp_layer = get_first_layer(learn.model, EmbeddingDropout)
    if emb_dp_name is not None:
        print('replace dropout layer')
        gnn_emb_dp = dhgnet.EmbeddingDropout(gnn_emb, config['embed_p'])
        set_layer(learn.model, emb_dp_name, gnn_emb_dp)
        replace_layers[emb_dp_layer] = gnn_emb_dp

    if is_fastai:
        update_cb = NeedUpdateCallback(learn, list(replace_layers.values()))
        learn.callbacks.append(update_cb)

        for i, layer_group in enumerate(learn.layer_groups):
            if layer_group in replace_layers:
                learn.layer_groups[i] = replace_layers[layer_group]
            for name, layer in layer_group.named_modules():
                if layer in replace_layers:
                    set_layer(layer_group, name, replace_layers[layer])

    return gnn_emb
