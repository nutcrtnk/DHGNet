from collections import defaultdict
import numpy as np
import torch
from torch_geometric.utils import add_self_loops

import source_embedding as S


def _to_w2w_lang(lang):
    if lang == 'zh':
        return 'zh_cn'
    return lang


class DHG:

    def __init__(self, dict_name='word2word', directed=True):
        self.directed = directed
        self.vocabs = defaultdict(dict)
        self.langs = []
        self.dictionary = defaultdict(dict)
        self.w2v_dict = {}
        self.dict_name = dict_name
        self.edges = None

    def add_lang(self, lang, w2v=None):

        if w2v is not None:
            print(f'load w2v: {w2v}, lang: {lang}')
            if w2v == 'rcsls':
                self.w2v_dict[lang] = S.FastTextAlignedEmbedding(lang)
            elif w2v == 'fasttext':
                self.w2v_dict[lang] = S.FastTextWikiEmbedding(lang)
            elif w2v.startswith('wk2v'):
                dim = int(w2v[4:])
                self.w2v_dict[lang] = S.Wikipedia2VecEmbedding(lang, dim)
            else:
                assert lang == 'en', 'invalid w2v'
                if w2v.startswith('glove'):
                    dim = int(w2v[5:])
                    self.w2v_dict[lang] = S.GloveEmbedding(
                        'wikipedia_gigaword', dim)
                elif w2v == 'crawl':
                    self.w2v_dict[lang] = S.GloveEmbedding('common_crawl_840')
                elif w2v == 'twitter':
                    self.w2v_dict[lang] = S.GloveEmbedding('twitter', 200)
                elif w2v == 'word2vec':
                    self.w2v_dict[lang] = S.Word2VecEmbedding()

        if len(self.langs) > 0:
            for lang2 in self.langs:
                if self.dict_name == 'word2word':
                    from word2word import Word2word
                    w2w_lang, w2w_lang2 = _to_w2w_lang(
                        lang), _to_w2w_lang(lang2)
                    self.dictionary[lang][lang2] = Word2word(
                        w2w_lang, w2w_lang2)
                    self.dictionary[lang2][lang] = Word2word(
                        w2w_lang2, w2w_lang)
        self.langs.append(lang)

    def _add_edges(self, source_lang, target_lang, dict_name=None):
        assert self.edges is not None, 'use this method via create_graph'
        lang, tlang = source_lang, target_lang
        if not self.directed:
            relation = f'{lang}-{tlang}' if lang < tlang else f'{tlang}-{lang}'
            rev_relation = relation
        else:
            relation = f'{lang}-{tlang}'
            rev_relation = f'{tlang}-{lang}'
        if dict_name is not None:
            relation += f'_{dict_name}'
            rev_relation += f'_{dict_name}'
        w2w = self.dictionary[source_lang][target_lang]

        for word, sword_id in sorted(self.vocabs[lang].items(), key=lambda x: x[1]):
            if word not in w2w.word2x:
                continue
            for tword in w2w(word):
                if (tword not in self.vocabs[tlang]) and (tlang in self.w2v_dict) and (tword in self.w2v_dict[tlang]):
                    self.vocabs[tlang][tword] = len(self.vocabs[tlang])
                if tword not in self.vocabs[tlang]:
                    continue
                tword_id = self.vocabs[tlang][tword]
                self.edges[tlang][lang][relation][tword_id][sword_id] = 1
                if not self.directed:
                    self.edges[lang][tlang][rev_relation][sword_id][tword_id] = 1

    def create_graph(self, to_first_only=True):
        self.edges = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(dict)
                )
            )
        )

        for lang in self.langs:
            for tlang in self.dictionary[lang]:
                if to_first_only and (lang != self.langs[0] and tlang != self.langs[0]):
                    continue
                self._add_edges(lang, tlang)

    def get_graph_data(self, self_loop=True, reverse=False):
        assert self.edges is not None, 'call create_graph before getting graph data'

        offset = []
        _offset = 0
        node_type = []
        for i, lang in enumerate(self.langs):
            offset.append(_offset)
            num_vocab = max(self.vocabs[lang].values()) + 1
            _offset += num_vocab
            node_type.append((torch.ones(num_vocab) * i).long())
        offset.append(_offset)
        node_type = torch.cat(node_type)

        relation2id = {}
        lang2id = {lang: i for i, lang in enumerate(self.langs)}
        relation_langs = []

        pairs = defaultdict(list)

        n_true_edges = 0
        for tlang in self.langs:
            t_offset = offset[lang2id[tlang]]
            for slang in sorted(self.edges[tlang].keys()):
                s_offset = offset[lang2id[slang]]
                for relation in sorted(self.edges[tlang][slang].keys()):
                    if relation not in relation2id:
                        relation2id[relation] = len(relation2id)
                        if reverse:
                            relation_langs.append(
                                (lang2id[tlang], lang2id[slang]))
                        else:
                            relation_langs.append(
                                (lang2id[slang], lang2id[tlang]))
                    rid = relation2id[relation]
                    for tword_id in self.edges[tlang][slang][relation]:
                        tid = tword_id + t_offset
                        for sword_id in self.edges[tlang][slang][relation][tword_id]:
                            sid = sword_id + s_offset
                            if reverse:
                                pairs[(tid, sid)].append(rid)
                            else:
                                pairs[(sid, tid)].append(rid)
                            n_true_edges += 1

        print(relation2id)
        edge_index = []
        edge_type = []

        def _add_edge(sid, tid, r_list, rev_list):
            for rid in r_list:
                edge_index.append([sid, tid])
                edge_type.append(rid)
            for rid in rev_list:
                edge_index.append([tid, sid])
                edge_type.append(rid)

        while len(pairs) > 0:
            pair, r_list = pairs.popitem()
            sid, tid = pair
            rev_list = pairs.pop((tid, sid)) if (tid, sid) in pairs else []
            _add_edge(sid, tid, r_list, rev_list)

        node_lang = {}
        for lang, stoi in self.vocabs.items():
            for i in stoi.values():
                node_lang[i + offset[lang2id[lang]]] = lang

        edge_index = torch.LongTensor(edge_index).t()
        edge_type = torch.LongTensor(edge_type)

        if self_loop:
            edge_index, edge_type = add_self_loops(
                edge_index, edge_weight=edge_type, fill_value=len(relation2id))
        offset = torch.LongTensor(offset)
        relation_langs = torch.LongTensor(relation_langs)
        return edge_index, edge_type, node_type, offset, relation_langs

    def get_pretrain(self):
        pretrains = []
        for lang in self.langs:
            if lang not in self.w2v_dict:
                pretrains.append(None)
                continue
            num_vocab = max(self.vocabs[lang].values()) + 1
            pretrain = None
            for w, wid in self.vocabs[lang].items():
                vec = torch.from_numpy(np.array(self.w2v_dict[lang][w]))
                if pretrain is None:
                    pretrain = vec.new_zeros([num_vocab, vec.numel()])
                pretrain[wid] = vec
            pretrains.append(pretrain)
        return pretrains

    def dump_w2v_temp(self):
        for w2v in self.w2v_dict.values():
            if isinstance(w2v, S.Embedding):
                w2v.dump_temp_query()
