import torch
from copy import deepcopy
from functools import partial
import torch.optim as optim
import numpy as np

from fastai.text.models import awd_lstm, AWD_LSTM
from fastai.metrics import accuracy, accuracy_thresh
from fastai.text.learner import text_classifier_learner, language_model_learner
from fastai.callbacks import SaveModelCallback, EarlyStoppingCallback
from fastai.basic_train import LearnerCallback
from fastai.metrics import MultiLabelFbeta

from utils import partialclass
import model as M
from dhgnet import DHGNet


class ModifiedMultiLabelFbeta(MultiLabelFbeta):

    def __init__(self, *args, multi_label=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_label = multi_label

    def _to_onehot(self, pred, targ):
        with torch.no_grad():
            onehot = torch.zeros_like(pred)
            onehot.scatter_(1, targ.data.unsqueeze(1), 1.)
        return onehot

    def on_batch_end(self, last_output, last_target, **kwargs):
        if last_target.dim() != last_output.dim():
            last_target = self._to_onehot(last_output, last_target)

        if self.multi_label:
            pred, targ = ((last_output.sigmoid(
            ) if self.sigmoid else last_output) > self.thresh).byte(), last_target.byte()
        else:
            last_output = last_output.argmax(dim=-1)
            last_output = self._to_onehot(last_target, last_output)
            pred, targ = last_output.byte(), last_target.byte()

        m = pred*targ
        self.tp += m.sum(0).float()
        self.total_pred += pred.sum(0).float()
        self.total_targ += targ.sum(0).float()


MicroF1 = partialclass(ModifiedMultiLabelFbeta,
                       'MicroF1', average='micro', beta=1)
MacroF1 = partialclass(ModifiedMultiLabelFbeta,
                       'MacroF1', average='macro', beta=1)


class EarlyStoppingCallbackFix(EarlyStoppingCallback):

    def on_epoch_end(self, epoch, **kwargs):
        "Compare the value monitored to its best score and maybe stop training."
        current = self.get_monitor_value()
        if isinstance(current, torch.Tensor):
            current = current.item()
        if current is None:
            return
        if self.operator(current - self.min_delta, self.best):
            self.best, self.wait = current, 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                print(f'Epoch {epoch}: early stopping')
                return {"stop_training": True}


class SaveModelCallbackFix(SaveModelCallback):

    def __init__(self, learn, monitor: str = 'valid_loss', mode: str = 'auto', every: str = 'improvement', name: str = 'bestmodel', reset_on_fit=False):
        super().__init__(learn, monitor=monitor, mode=mode, every=every, name=name)
        self.best = float('inf') if self.operator == np.less else -float('inf')
        self.reset_on_fit = reset_on_fit

    def on_train_begin(self, **kwargs):
        "Initializes the best value."
        if self.reset_on_fit:
            self.best = float(
                'inf') if self.operator == np.less else -float('inf')


def get_model_name(args):
    return f"{args.name}"


def set_lstm_config(args, is_lm=False):
    config = deepcopy(
        awd_lstm.awd_lstm_lm_config if is_lm else awd_lstm.awd_lstm_clas_config)
    config['emb_sz'] = args.emb_sz
    config['n_layers'] = args.rnn_layers
    return config


def train_aligned(gnn_emb: DHGNet, epochs):
    batch_size = 50000
    print('Train aligned')
    opt = optim.Adam(gnn_emb.parameters(), lr=1e-3)
    gnn_emb.train()
    for i in range(epochs):
        loss = gnn_emb.aligned_loss(batch_size=batch_size)
        print(f'epoch {i} - loss: {loss.item()}')
        loss.backward()
        opt.step()
        opt.zero_grad()
    del opt


def train_lm(data_lm, args, print_model=False):

    model_name = get_model_name(args)
    model_arch = AWD_LSTM
    config = set_lstm_config(args, is_lm=True)
    config['tie_weights'] = args.tie_weights
    trn_args = dict(drop_mult=0.9, clip=0.12, alpha=2, beta=1)

    learn = language_model_learner(
        data_lm, model_arch, config=config, pretrained=False, **trn_args)
    gnn_emb = M.modify_language_model(args, learn, data_lm.vocab, config)

    if args.pretrain_lm_epochs:
        monitor = 'valid_loss'
        callback_kwargs = dict(callbacks=[SaveModelCallbackFix(
            learn, every='improvement', name=f'{model_name}_lm', monitor=monitor, reset_on_fit=True), EarlyStoppingCallbackFix(learn, patience=5, monitor=monitor)])
        callback_kwargs['callbacks'].append(
            LMCallback(learn, args.num_lm_vocab))

        learn = learn.to_fp32()
        print('learner done')

        if print_model:
            print(learn.model)

        if args.pretrain_align_epochs:
            train_aligned(gnn_emb, args.pretrain_align_epochs)

        print('training unfrozen')
        learn.unfreeze()
        epochs = args.pretrain_lm_epochs
        n_groups = len(learn.layer_groups) - 1
        lr = slice(1e-3 / (2.6 ** n_groups), 1e-3)
        learn.fit(epochs, lr, **callback_kwargs)

        learn.load(f'{model_name}_lm')
        learn.save_encoder(f'{model_name}_enc')
        print('done training lm')

        model_path = learn.path / learn.model_dir / f'{model_name}_lm.pth'
        model_path.unlink()

    return learn


def train_cls(data_cls, args, print_model=False):

    model_name = get_model_name(args)
    model_arch = AWD_LSTM
    config = set_lstm_config(args)
    metrics = [accuracy_thresh if args.multi_label else accuracy, MicroF1(
        thresh=0.5, multi_label=args.multi_label), MacroF1(thresh=0.5, multi_label=args.multi_label)]
    monitor = 'micro_f1'

    trn_args = dict(bptt=70, max_len=700, drop_mult=0.5,
                    alpha=2, beta=1, metrics=metrics)

    learn = text_classifier_learner(
        data_cls, model_arch, config=config, pretrained=False, **trn_args)
    gnn_emb = M.modify_classification_model(
        args, learn, data_cls.vocab, config)

    learn = learn.to_fp32()

    if args.load_lm_gnn:
        learn.load_encoder(args.load_lm_gnn)
    elif args.pretrain_lm_epochs:
        learn.load_encoder(f'{model_name}_enc')

    callback_kwargs = dict(callbacks=[SaveModelCallbackFix(learn, every='improvement', monitor=monitor,
                           name=f'{model_name}_cls'), EarlyStoppingCallbackFix(learn, monitor=monitor, patience=5)])

    if print_model:
        print(learn.model)

    n_groups = len(learn.layer_groups) - 1
    epochs = args.train_cls_epochs

    if epochs < 0:
        print('load trained classifier')
        learn.load(f'{model_name}_cls')
        return learn

    if args.pretrain_lm_epochs or args.load_lm_gnn:

        learn.opt_func = partial(optim.Adam, betas=(0.7, 0.99))

        print('freeze -1')
        learn.freeze_to(-1)
        learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))

        print('freeze -2')
        if n_groups >= 2:
            learn.freeze_to(-2)
            learn.fit_one_cycle(3, slice(1e-2 / (2.6 ** n_groups), 1e-2),
                                moms=(0.8, 0.7), **callback_kwargs)
            learn.load(f'{model_name}_cls')

        if n_groups >= 3:
            print('freeze -3')
            learn.freeze_to(-3)
            learn.fit_one_cycle(3, slice(5e-3 / (2.6 ** n_groups), 5e-3),
                                moms=(0.8, 0.7), **callback_kwargs)
            learn.load(f'{model_name}_cls')

        print('unfreeze')
        torch.cuda.empty_cache()
        learn.unfreeze()
        learn.fit_one_cycle(epochs, slice(1e-3 / (2.6 ** n_groups), 1e-3),
                            moms=(0.8, 0.7), **callback_kwargs)

    else:
        if args.pretrain_align_epochs:
            train_aligned(gnn_emb, args)

        learn.opt_func = partial(optim.Adam, betas=(0.9, 0.99))
        max_lr = 1e-3

        learn.unfreeze()
        lr = slice(max_lr / (2.6 ** n_groups), max_lr)
        learn.fit(epochs, lr, **callback_kwargs)

    learn.load(f'{model_name}_cls')
    return learn


class LMCallback(LearnerCallback):

    def __init__(self, learn, num_vocab):
        super().__init__(learn)
        print(f'limit LM training vocab to {num_vocab}')
        self.num_vocab = num_vocab

    def on_loss_begin(self, last_output, **kwargs):
        return {'last_output': last_output[..., :self.num_vocab]}
