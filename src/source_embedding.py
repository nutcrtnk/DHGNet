from gensim.utils import pickle, unpickle
from collections import namedtuple
from os import path
from pathlib import Path
import zipfile
from tqdm import tqdm
from gensim.models import KeyedVectors
from array import array
import sys
import time
import os
import requests

from embeddings.embedding import Embedding as Embedding_

w2v_folder = Path('../data/word_emb')


def download_url(url: str, dest: str, overwrite: bool = False,
                 show_progress=True, chunk_size=1024*1024, timeout=4, retries=5) -> None:
    "Download `url` to `dest` unless it exists and not `overwrite`."
    if os.path.exists(dest) and not overwrite:
        return

    s = requests.Session()
    s.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
    u = s.get(url, stream=True, timeout=timeout)
    try:
        file_size = int(u.headers["Content-Length"])
    except:
        show_progress = False

    with open(dest, 'wb') as f:
        nbytes = 0
        if show_progress:
            pbar = tqdm(range(file_size))
        try:
            for chunk in u.iter_content(chunk_size=chunk_size):
                nbytes += len(chunk)
                if show_progress:
                    pbar.update(nbytes)
                f.write(chunk)
        except requests.exceptions.ConnectionError as e:
            timeout_txt = (f'\n Download of {url} has failed after {retries} retries\n'
                           f' Fix the download manually:\n')
            print(timeout_txt)
            import sys
            sys.exit(1)


class Embedding(Embedding_):

    def __init__(self, use_temp_only=False) -> None:
        super().__init__()
        self._temp_query = None
        self.db_path = None
        self.use_temp_only = use_temp_only

    @staticmethod
    def path(p):
        return str(w2v_folder / p)

    def __getitem__(self, w):
        return self.lookup(w)

    def lookup(self, w):
        if self._temp_query is None:
            self.load_temp_query()
        if w in self._temp_query:
            return self._temp_query[w]
        if self.use_temp_only:
            return None

        c = self.db.cursor()
        q = c.execute('select emb from embeddings where word = :word', {
                      'word': w}).fetchone()
        res = array('f', q[0]).tolist() if q else None
        self._temp_query[w] = res
        return res

    def load_temp_query(self):
        if self.db_path is None:
            self._temp_query = {}
            return
        pkl_path = self.path(self.db_path + '_temp.pkl')
        if path.exists(pkl_path):
            print(f'load temp query from {pkl_path}')
            n_try = 0
            while True:
                try:
                    self._temp_query = unpickle(pkl_path)
                    return
                except:
                    n_try += 1
                    if n_try > 10:
                        e = sys.exc_info()[0]
                        raise Exception(str(e))
                    print(f'try to unpickle #{n_try}')
                    time.sleep(2)
        else:
            self._temp_query = {}

    def dump_temp_query(self):
        if self.db_path is None:
            print('cannot dump temp without db_path!')
            return
        if self._temp_query is not None:
            print('dump temp query')
            pkl_path = self.path(self.db_path + '_temp.pkl')
            n_try = 0
            while True:
                try:
                    pickle(self._temp_query, pkl_path)
                    return
                except:
                    n_try += 1
                    if n_try > 10:
                        e = sys.exc_info()[0]
                        raise Exception(str(e))
                    print(f'try to pickle #{n_try}')
                    time.sleep(2)

    def _load_w2v_txt(self, w2v_path, has_header=True, batch_size=1000):
        batch = []
        seen = set()
        with open(w2v_path, 'r', encoding='utf-8') as fin:
            if has_header:
                n, d = map(int, next(fin).split())
            data = {}
            for line in fin:
                elems = line.strip().split(' ')
                vec = [float(n) for n in elems[1:]]
                word = elems[0]
                if word in seen:
                    continue
                seen.add(word)
                batch.append((word, vec))
                if len(batch) == batch_size:
                    self.insert_batch(batch)
                    batch.clear()
            if batch:
                self.insert_batch(batch)
        return data

    def _load_gensim_w2v(self, w2v_path, batch_size=1000, binary=False):
        batch = []
        word2vec = KeyedVectors.load_word2vec_format(w2v_path, binary=binary)
        for word in word2vec.vocab:
            batch.append((word, word2vec[word]))
            if len(batch) == batch_size:
                self.insert_batch(batch)
                batch.clear()
        if batch:
            self.insert_batch(batch)


class FastTextWikiEmbedding(Embedding):
    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'
    w2v_file = 'wiki.{}.vec'
    pretrain = 'fasttext_wiki'
    """
    Reference: https://arxiv.org/abs/1607.04606
    """

    def __init__(self, lang='en', **kwargs):
        super().__init__(**kwargs)
        self.lang = lang
        db_path = self.path(path.join(self.pretrain, '{}.db'.format(lang)))
        self.db_path = db_path
        if not self.use_temp_only:
            is_load = not path.exists(db_path)
            self.db = self.initialize_db(db_path)
            if is_load or path.getsize(db_path) < 1000000:
                self.clear()
                self.load_word2emb(self.lang)

    def load_word2emb(self, lang):
        dest = self.path(self.w2v_file.format(lang))
        download_url(self.url.format(lang), dest=dest)
        self._load_w2v_txt(dest)


class FastTextAlignedEmbedding(FastTextWikiEmbedding):
    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.{}.align.vec'
    w2v_file = 'wiki.{}.align.vec'
    pretrain = 'fasttext_aligned'


class GloveEmbedding(Embedding):

    GloveSetting = namedtuple(
        'GloveSetting', ['url', 'd_embs', 'size', 'description'])
    settings = {
        'common_crawl_48': GloveSetting('http://nlp.stanford.edu/data/glove.42B.300d.zip',
                                        [300], 1917494, '48B token common crawl'),
        'common_crawl_840': GloveSetting('http://nlp.stanford.edu/data/glove.840B.300d.zip',
                                         [300], 2195895, '840B token common crawl'),
        'twitter': GloveSetting('http://nlp.stanford.edu/data/glove.twitter.27B.zip',
                                [25, 50, 100, 200], 1193514, '27B token twitter'),
        'wikipedia_gigaword': GloveSetting('http://nlp.stanford.edu/data/glove.6B.zip',
                                           [50, 100, 200, 300], 400000, '6B token wikipedia 2014 + gigaword 5'),
    }

    @staticmethod
    def path(p):
        return str(w2v_folder / p)

    def __init__(self, name='common_crawl_840', d_emb=300, show_progress=True, **kwargs):
        """

        Args:
            name: name of the embedding to retrieve.
            d_emb: embedding dimensions.
            show_progress: whether to print progress.
        """
        super().__init__(**kwargs)
        assert name in self.settings, '{} is not a valid corpus. Valid options: {}'.format(
            name, self.settings)
        self.setting = self.settings[name]
        assert d_emb in self.setting.d_embs, '{} is not a valid dimension for {}. Valid options: {}'.format(
            d_emb, name, self.setting)

        self.d_emb = d_emb
        self.name = name
        db_path = self.path(path.join('glove', '{}:{}.db'.format(name, d_emb)))
        self.db_path = db_path
        if not self.use_temp_only:
            is_load = not path.exists(db_path)
            self.db = self.initialize_db(db_path)
            if is_load or path.getsize(db_path) < 1000000:
                self.clear()
                self.load_word2emb(show_progress=show_progress)

    def load_word2emb(self, show_progress=True, batch_size=1000):
        fin_name = self.path(path.join('glove', '{}.zip'.format(self.name)))
        download_url(self.setting.url, dest=fin_name)
        # fin_name = self.ensure_file(path.join('glove', '{}.zip'.format(self.name)), url=self.setting.url)
        seen = set()
        with zipfile.ZipFile(fin_name) as fin:
            fname_zipped = [fzipped.filename for fzipped in fin.filelist if str(
                self.d_emb) in fzipped.filename][0]
            with fin.open(fname_zipped, 'r') as fin_zipped:
                batch = []
                if show_progress:
                    fin_zipped = tqdm(fin_zipped, total=self.setting.size)
                for line in fin_zipped:
                    elems = line.decode().rstrip().split()
                    vec = [float(n) for n in elems[-self.d_emb:]]
                    word = ' '.join(elems[:-self.d_emb])
                    if word in seen:
                        continue
                    seen.add(word)
                    batch.append((word, vec))
                    if len(batch) == batch_size:
                        self.insert_batch(batch)
                        batch.clear()
                if batch:
                    self.insert_batch(batch)


class Word2VecEmbedding(Embedding):
    w2v_file = 'GoogleNews-vectors-negative300.bin.gz'
    pretrain = 'word2vec'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim = 300
        db_path = self.path(path.join(self.pretrain, '{}.db'.format(dim)))
        self.db_path = db_path
        if not self.use_temp_only:
            is_load = not path.exists(db_path)
            self.db = self.initialize_db(db_path)
            if is_load or path.getsize(db_path) < 1000000:
                self.clear()
                self.load_word2emb()

    def load_word2emb(self):
        dest = self.path(self.w2v_file)
        self._load_gensim_w2v(dest, binary=True)


class Wikipedia2VecEmbedding(Embedding):
    url = 'http://wikipedia2vec.s3.amazonaws.com/models/{}/2018-04-20/{}wiki_20180420_{}d.txt.bz2'
    w2v_file = 'wiki2vec_{}_{}.txt.bz2'
    pretrain = 'wikipedia2vec'

    def __init__(self, lang='en', dim=300, **kwargs):
        super().__init__(**kwargs)
        self.lang = lang
        self.dim = dim
        db_path = self.path(
            path.join(self.pretrain, '{}_{}.db'.format(lang, dim)))
        self.db_path = db_path
        if not self.use_temp_only:
            is_load = not path.exists(db_path)
            self.db = self.initialize_db(db_path)
            if is_load or path.getsize(db_path) < 1000000:
                self.clear()
                self.load_word2emb(self.lang, self.dim)

    def load_word2emb(self, lang, dim):
        dest = self.path(self.w2v_file.format(lang, dim))
        download_url(self.url.format(lang, lang, dim), dest=dest)
        self._load_gensim_w2v(dest)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('lang')
    parser.add_argument('dim', type=int)
    args = parser.parse_args()

    if args.name == 'align':
        w2v = FastTextAlignedEmbedding(args.lang)
    elif args.name == 'wiki':
        w2v = FastTextWikiEmbedding(args.lang)
    elif args.name == 'glove':
        w2v = GloveEmbedding('wikipedia_gigaword', args.dim)
    elif args.name == 'wk2v':
        w2v = Wikipedia2VecEmbedding(args.lang, args.dim)
    else:
        raise Exception('invalid name')
    print('Finish loading')
    print(f'#embs {len(w2v)}')
