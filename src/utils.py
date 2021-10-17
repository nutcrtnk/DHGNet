from pathlib import Path
from copy import deepcopy


def partialclass(cls, new_name, *args, **kwds):
    class NewCls(cls):
        def __init__(self, *cls_args, **cls_kwds):
            cls_args = args + tuple(cls_args)
            cls_kwds.update(kwds)
            super().__init__(*cls_args, **cls_kwds)
    NewCls.__name__ = new_name
    return NewCls

def copy_vocab(vocab):
    vocab_cpy = deepcopy(vocab)
    vocab_cpy.stoi = deepcopy(vocab.stoi)
    return vocab_cpy
