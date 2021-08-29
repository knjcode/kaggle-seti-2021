import math
import time
import torch
import pandas as pd

from argparse import ArgumentParser

from sklearn.model_selection import StratifiedKFold


def bn_to_syncbn(module):
    from horovod.torch.sync_batch_norm import SyncBatchNorm
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module = SyncBatchNorm(module.num_features, eps=module.eps, momentum=module.momentum, affine=module.affine, track_running_stats=module.track_running_stats)


class DummySummaryWriter:
    def __init__(*args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, *args, **kwargs):
        return self


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--conf', required=True,
                        help='Path to config file')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--new_train', action='store_true')
    parser.add_argument('--new_test', action='store_true')
    parser.add_argument('-m', '--ckpt_path', type=str, default=None)
    parser.add_argument('--input_width', type=int, default=None)
    parser.add_argument('--input_height', type=int, default=None)
    parser.add_argument('--scale_width', type=int, default=None)
    parser.add_argument('--scale_height', type=int, default=None)
    parser.add_argument('--valid_bs', type=int, default=None)
    parser.add_argument('--valid_trans_mode', type=str, default=None)
    parser.add_argument('--train_fold', type=str, default='')
    parser.add_argument('--overwrite_gem_p', type=float, default=None)
    parser.add_argument('--ensemble_sigmoid', action='store_true')
    parser.add_argument('--tta_hflip', action='store_true')
    parser.add_argument('--tta_vflip', action='store_true')
    parser.add_argument('--tta_sigmoid', action='store_true')
    parser.add_argument('--seed', type=int, default=None)

    return parser.parse_args()


def get_folds(conf, N_FOLDS, FOLD_SEED, DATA, logger):
    if N_FOLDS == 5:
        train = pd.read_csv('train_labels_5fold.csv')
    else:
        train = pd.read_csv(DATA / "train_labels.csv")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
        train["fold"] = -1
        for fold_id, (_, val_idx) in enumerate(skf.split(train["id"], train["target"])):
            train.loc[val_idx, "fold"] = fold_id

    train.groupby("fold").agg(total=("id", len), pos=("target", sum))
    return train
