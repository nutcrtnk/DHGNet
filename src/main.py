from utils import copy_vocab
import run_setting
from train import train_lm, train_cls, get_model_name
from nltk import ToktokTokenizer as ToktokTokenizer_
from fastai.text.data import NumericalizeProcessor, TextList, ItemLists, TokenizeProcessor
from fastai.basic_data import DatasetType
from fastai.text.transform import Tokenizer, BaseTokenizer
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
import argparse

data_folder = Path('../data/text_cls')


class ToktokTokenizer(BaseTokenizer):
    "Basic class for a tokenizer function."

    def __init__(self, lang: str):
        self.lang = lang
        self.base = ToktokTokenizer_()

    def tokenizer(self, t: str):
        return self.base.tokenize(t)


def get_tokenizer(lang):
    if lang == 'th':
        from pythainlp.ulmfit import ThaiTokenizer, pre_rules_th, post_rules_th
        return Tokenizer(tok_func=ThaiTokenizer, lang='th', pre_rules=pre_rules_th, post_rules=post_rules_th)
    else:
        return Tokenizer(tok_func=ToktokTokenizer, lang=lang)


def get_tokenizer_preprocesser(args):
    tokenizer = get_tokenizer(args.target_lang)
    return TokenizeProcessor(tokenizer=tokenizer, chunksize=10000, mark_fields=False)


def eval_multiclass(preds, y_true):
    preds = np.argmax(preds, axis=-1)
    f1_macro = f1_score(y_true, preds, average='macro')
    acc = accuracy_score(y_true, preds)
    scores = dict(F1_macro=f1_macro, Acc=acc)
    return scores


def eval_multilabel(pred_probs, y_true):
    preds = (pred_probs > 0.5).astype(int)
    f1_macro = f1_score(y_true, preds, average='macro')
    acc = accuracy_score(y_true, preds)
    scores = dict(F1_macro=f1_macro, Acc=acc)
    return scores


def print_scores(scores):
    str_out = ', '.join(
        f'{metric}: {value:.4f}' for metric, value in scores.items())
    print(str_out)


def print_scores_summary(scores):
    str_out = ', '.join(
        f'{metric}: {np.mean(values):.4f}' for metric, values in scores.items())
    print('Mean:', str_out)
    str_out = ', '.join(
        f'{metric}: {np.std(values):.4f}' for metric, values in scores.items())
    print('Std :', str_out)


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def none_or_str(s):
    if s is None or s.lower() == 'none':
        return None
    return s


def none_or_int(s):
    if s is None or s.lower() == 'none':
        return None
    return int(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=str)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--runs', type=int, default=5)

    # DHG arguments
    parser.add_argument('--langs', type=str, default='en')
    parser.add_argument('--dict', type=str, default='word2word')
    parser.add_argument('--directed', type=bool_flag, default=True)
    parser.add_argument('--w2v', type=none_or_str, default='fasttext')
    parser.add_argument('--reverse', type=bool_flag, default=False)
    parser.add_argument('--add_from_dict', type=int, default=0)
    parser.add_argument('--save_temp', type=bool_flag, default=True)
    parser.add_argument('--use_temp_only', type=bool_flag, default=False)

    # DHGNet arguments
    parser.add_argument('--conv', type=str, default='hgnn')
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--layer_norm', type=bool_flag, default=True)
    parser.add_argument('--residual', type=str, default=True)
    parser.add_argument('--freeze_cross', type=bool_flag, default=True)

    # Classifier->AWDLSTM arguments
    parser.add_argument('--emb_sz', type=int, default=300)
    parser.add_argument('--rnn_layers', type=int, default=3)
    parser.add_argument('--tie_weights', type=bool_flag, default=True)

    # Training arguments
    parser.add_argument('--pretrain_align_epochs', type=int, default=50)
    parser.add_argument('--pretrain_lm_epochs', type=int, default=200)
    parser.add_argument('--train_cls_epochs', type=int, default=20)
    parser.add_argument('--load_lm_gnn', type=none_or_str, default=None)

    args = parser.parse_args()
    args = run_setting.update(args)
    print(args)

    assert args.pretrain_lm_epochs or args.train_cls_epochs

    model_name = get_model_name(args)
    folder = data_folder / f'{args.dataset}'
    model_path = folder

    train_df = pd.read_csv(folder / 'train_df.csv')
    valid_df = pd.read_csv(folder / 'valid_df.csv')
    test_df = pd.read_csv(folder / 'test_df.csv')

    if args.label_col is None and args.multi_label:
        args.label_col = list(train_df.columns[1:])

    text_col = args.text_col
    label_col = args.label_col

    vocab = None
    vocab_path = folder / 'models' / f'{model_name}_vocab.pkl'
    vocab_path.parent.mkdir(exist_ok=True)

    # prepare vocab from dataset
    data_prep = TextList.from_df(train_df, model_path, cols=[text_col], processor=[
                                 get_tokenizer_preprocesser(args), NumericalizeProcessor(vocab=vocab)])
    data_prep = data_prep.process()
    vocab = data_prep.vocab

    args.num_lm_vocab = len(vocab.itos)
    del data_prep

    is_print_model = True
    numericalizer = NumericalizeProcessor(vocab=copy_vocab(vocab))

    def _train_lm():
        nonlocal is_print_model, numericalizer
        if numericalizer is None:
            numericalizer = NumericalizeProcessor(vocab=copy_vocab(vocab))
        processor = [get_tokenizer_preprocesser(args), numericalizer]
        train_data = TextList.from_df(train_df, model_path, cols=[
                                      text_col], processor=processor)
        val_data = TextList.from_df(valid_df, model_path, cols=[
                                    text_col], processor=processor)
        data_lm = (ItemLists(model_path, train=train_data, valid=val_data)
                   .label_for_lm()
                   .databunch(bs=64)
                   )
        train_lm(data_lm, args, print_model=is_print_model)
        data_lm.vocab.save(vocab_path)
        is_print_model = False
        del data_lm
        torch.cuda.empty_cache()

    def _train_cls():
        nonlocal is_print_model, numericalizer
        if numericalizer is None:
            numericalizer = NumericalizeProcessor(vocab=copy_vocab(vocab))

        processor = [get_tokenizer_preprocesser(args), numericalizer]
        train_data = TextList.from_df(train_df, model_path, cols=[
                                      text_col], processor=processor)
        val_data = TextList.from_df(valid_df, model_path, cols=[
                                    text_col], processor=processor)
        data_cls = (ItemLists(model_path, train=train_data, valid=val_data)
                    .label_from_df(label_col, label_cls=None)
                    .add_test(TextList.from_df(test_df, model_path, cols=[text_col], processor=processor))
                    .databunch(bs=32)
                    )
        learn = train_cls(data_cls, args, print_model=is_print_model)
        is_print_model = False
        return learn

    if args.pretrain_lm_epochs > 0:
        _train_lm()
    if args.train_cls_epochs <= 0:
        return

    valid_scores = defaultdict(list)
    test_scores = defaultdict(list)

    for i_run in range(args.runs):
        print(f'===== RUN #{i_run} =====')
        learn = _train_cls()

        for _m in learn.model.modules():
            if hasattr(_m, '_need_update'):
                _m._need_update = True

        y_val, y_test = np.array(
            valid_df[label_col]), np.array(test_df[label_col])
        if not args.multi_label:
            y_val, y_test = np.vectorize(learn.data.c2i.get)(
                y_val), np.vectorize(learn.data.c2i.get)(y_test)
        probs, y = learn.get_preds(DatasetType.Valid, ordered=True)
        preds = probs.numpy()
        if not args.multi_label:
            scores = eval_multiclass(preds, y_val)
        else:
            scores = eval_multilabel(preds, y_val)
        print_scores(scores)
        for metric, value in scores.items():
            valid_scores[metric].append(value)

        probs, y = learn.get_preds(DatasetType.Test, ordered=True)
        preds = probs.numpy()
        if not args.multi_label:
            scores = eval_multiclass(preds, y_test)
        else:
            scores = eval_multilabel(preds, y_test)

        print_scores(scores)
        for metric, value in scores.items():
            test_scores[metric].append(value)

        del learn
        torch.cuda.empty_cache()

        print('===== END RUN =====')
        print()

    print('Valid')
    print_scores_summary(valid_scores)
    print('Test')
    print_scores_summary(test_scores)


if __name__ == '__main__':
    main()
