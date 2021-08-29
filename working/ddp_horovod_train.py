import datetime
import logging
import pytz
import os
import random
import re
import time

from pathlib import Path

import logzero
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from logzero import logger
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import get_model
from dataset import SetiSimpleDataset, get_transforms
from config import load_config
from util import parse_args, get_folds
from optimizer import get_optimizer
from scheduler import get_scheduler
from module import train_fn, valid_fn, mixup_train_fn
from loss import FocalLoss

import torch.backends.cudnn as cudnn
import horovod.torch as hvd
import torch.utils.data.distributed

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from util import bn_to_syncbn, DummySummaryWriter

# Some of the code was adapted from the following URL
# https://www.kaggle.com/yasufuminakama/cassava-resnext50-32x4d-starter-training

ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "sbl"
TRAIN = DATA / "train"
TEST = DATA / "test"


def get_path_label(df: pd.DataFrame, img_dir: str):
    """Get file path and target info."""
    path_label = {
        "paths": [img_dir / f"{img_id[0]}/{img_id}.npy" for img_id in df["id"].values],
        "labels": df[CLASSES].values.astype("f"),
        "id": df["id"].values
    }

    return path_label


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class ReduceLROnPlateauPatch(ReduceLROnPlateau):
    def get_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]


def train_loop(conf, hvd, folds, fold, logger, log_basename, total_epochs,
               new_train, new_test):

    logger.info(f"=============== fold: {fold} training ===============")

    if conf.ckpt_path:
        conf.ckpt_path = re.sub('fold._best', f"fold{fold}_best", conf.ckpt_path)
        logger.info(f"replace ckpt_path: {conf.ckpt_path}")

    # foldsをidでソートしておく(ddpのvalidationのため)
    folds = folds.sort_values(by=['id']).reset_index(drop=True)

    # loader
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    if new_train:
        valid_folds = pd.read_csv(DATA / 'train_labels.csv')
    if new_test:
        valid_folds = pd.read_csv(DATA / 'sample_submission.csv')

    tb_logname = os.path.join(conf.log_dir, f"{log_basename}_fold{fold}")
    if hvd.rank() == 0:
        tb_writer = SummaryWriter(log_dir=tb_logname)
    else:
        tb_writer = DummySummaryWriter()

    train_path_label = get_path_label(train_folds, TRAIN)
    valid_path_label = get_path_label(valid_folds, TRAIN)

    if new_train:
        valid_path_label = get_path_label(valid_folds, TRAIN)
    if new_test:
        valid_path_label = get_path_label(valid_folds, TEST)

    # pseudo label
    if conf.pseudo_label:
        pseudo_folds = pd.read_csv('pseudo_test_labels.csv')
        pseudo_path_label = get_path_label(pseudo_folds, TEST)

        train_path_label['paths'] = np.concatenate([train_path_label['paths'], pseudo_path_label['paths']])
        train_path_label['labels'] = np.concatenate([train_path_label['labels'], pseudo_path_label['labels']])
        train_path_label['id'] = np.concatenate([train_path_label['id'], pseudo_path_label['id']])
        logger.info("use pseudo labeled data")


    train_dataset = SetiSimpleDataset(paths=train_path_label['paths'],
                                      labels=train_path_label['labels'],
                                      ids=train_path_label['id'],
                                      transform=get_transforms(conf, conf.train_trans_mode),
                                      target_only=conf.target_only,
                                      seed=conf.seed)
    valid_dataset = SetiSimpleDataset(paths=valid_path_label['paths'],
                                      labels=valid_path_label['labels'],
                                      ids=valid_path_label['id'],
                                      transform=get_transforms(conf, conf.valid_trans_mode),
                                      target_only=conf.target_only,
                                      with_id=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = DataLoader(train_dataset,
                              batch_size=conf.train_bs,
                              sampler=train_sampler,
                              num_workers=conf.num_workers,
                              pin_memory=conf.pin_memory,
                              drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=conf.valid_bs,
                              sampler=valid_sampler,
                              num_workers=conf.num_workers,
                              pin_memory=conf.pin_memory,
                              drop_last=False)

    if conf.mixup:
        # gen second train_dataset
        train_dataset2 = SetiSimpleDataset(paths=train_path_label['paths'],
                                           labels=train_path_label['labels'],
                                           ids=train_path_label['id'],
                                           transform=get_transforms(conf, conf.train_trans_mode),
                                           target_only=conf.target_only,
                                           seed=conf.seed+1000)
        train_sampler2 = torch.utils.data.distributed.DistributedSampler(
            train_dataset2, num_replicas=hvd.size(), rank=hvd.rank())
        train_loader2 = DataLoader(train_dataset2,
                                   batch_size=conf.train_bs,
                                   sampler=train_sampler2,
                                   num_workers=conf.num_workers,
                                   pin_memory=conf.pin_memory,
                                   drop_last=True)

    # update print_freq
    if conf.print_freq == 0:
        conf.print_freq = max(2, len(train_loader) // 10)

    # model
    device = torch.device(conf.device)

    if conf.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif conf.loss_type == 'focal':
        criterion = FocalLoss(gamma=conf.focal_loss_gamma)
    else:
        raise NotImplementedError(conf.loss_type)

    model = get_model(conf, conf.backbone_model_name, logger)

    if conf.sync_bn:
        model.apply(bn_to_syncbn)
        logger.info('convert bn to sync_bn')

    if conf.overwrite_gem_p:
        model.global_pool.p.data.fill_(conf.overwrite_gem_p)
        logger.info(f"overwrite_gem_p: {conf.overwrite_gem_p}")

    model = model.to(device)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    parameters = [
        {'params': model.parameters()}
    ]
    named_parameters = list(model.named_parameters())

    optimizer = get_optimizer(conf, parameters)
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=named_parameters,
        compression=hvd.Compression.none)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    scheduler = get_scheduler(conf, optimizer, train_loader, logger)

    global_step = 0

    best_score = 0
    best_loss = np.inf

    current_lr = scheduler.get_last_lr()
    logger.info(f"lr: {current_lr}")
    tb_writer.add_scalar('Other/LearningRate', current_lr[0], global_step)

    best_y_true = None
    best_y_preds = None

    for epoch in range(conf.epochs):

        start_time = time.time()

        if conf.train:
            if conf.mixup:
                avg_loss, global_step = mixup_train_fn(conf, global_step,
                                                       train_loader, train_loader2,
                                                       model, criterion,
                                                       optimizer, epoch, scheduler, device, train_sampler, train_sampler2, logger, tb_writer)
            else:
                avg_loss, global_step = train_fn(conf, global_step, train_loader, model, criterion,
                                                 optimizer, epoch, scheduler, device, train_sampler, logger, tb_writer)

        # val
        avg_loss, score, y_true, y_preds = valid_fn(conf, global_step, valid_loader, model, criterion,
                                                    device, True, hvd, logger, tb_writer, new_test)

        if isinstance(scheduler, ReduceLROnPlateau):
            if conf.plateau_mode == 'min':
                scheduler.step(avg_loss)
            elif conf.plateau_mode == 'max':
                scheduler.step(score)

        current_lr = scheduler.get_last_lr()
        logger.info(f"lr: {current_lr}")
        tb_writer.add_scalar('Other/LearningRate', current_lr[0], global_step)

        if conf.train:
            if score > best_score:
                best_score = score
                logger.info(f'Fold {fold} Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                if hvd.rank() == 0:
                    torch.save({'model': model.state_dict()},
                                f'{OUTPUT_DIR}/fold{fold}_best_score.pth')
                best_y_true = y_true
                best_y_preds = y_preds
            if avg_loss < best_loss:
                best_loss = avg_loss
                logger.info(f'Fold {fold} Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
                if hvd.rank() == 0:
                    torch.save({'model': model.state_dict()},
                                f'{OUTPUT_DIR}/fold{fold}_best_loss.pth')

            if hvd.rank() == 0:
                torch.save({'model': model.state_dict()},
                            f'{OUTPUT_DIR}/fold{fold}_epoch{epoch}.pth')
        else:
            if score > best_score:
                best_score = score
                best_y_true = y_true
                best_y_preds = y_preds

        elapsed = time.time() - start_time

        if conf.train:
            logger.info(f'Fold {fold} Epoch {epoch+1} - AUROC score: {score:.4f} Best: {best_score:.4f} time: {elapsed:.0f}s')
            logger.info(f'output_dir: {OUTPUT_DIR}')
        else:
            logger.info(f'AUROC score: {score:.4f}')

        total_epochs -= 1
        full_elapsed = total_epochs * int(elapsed)
        time_delta = datetime.timedelta(seconds=full_elapsed)
        now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        end_time = now + time_delta
        logger.info(f"Expected remaining seconds: {full_elapsed} sec")
        logger.info(f"Expected end time: {end_time}")

    valid_folds['preds'] = best_y_preds

    return best_score, best_y_true, best_y_preds, valid_folds


if __name__ == '__main__':
    args = parse_args()
    conf = load_config(args.conf)

    for (key, value) in args._get_kwargs():
        if key in ['input_width', 'input_height',
                   'scale_width', 'scale_height',
                   'valid_bs', 'valid_trans_mode',
                   'ckpt_path', 'train_fold',
                   'overwrite_gem_p',
                   'tta_hflip', 'tta_vflip', 'tta_sigmoid',
                   'seed']:
            if value:
                setattr(conf, key, value)

    if args.test or args.new_train or args.new_test:
        conf.train = False
        conf.epochs = 1

    seed_everything(conf.seed)

    hvd.init()
    torch.manual_seed(conf.seed)

    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(conf.seed)
    cudnn.benchmark = True

    FOLD_SEED = conf.fold_seed
    CLASSES = ["target",]
    N_FOLDS = conf.n_fold

    formatter = logging.Formatter('%(message)s')
    logzero.formatter(formatter)

    if not os.path.exists(conf.log_dir):
        os.makedirs(conf.log_dir, exist_ok=True)

    if args.new_train:
        log_basename = f"new_train_{conf.prefix}-{conf.backbone_model_name}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    elif args.new_test:
        log_basename = f"new_test_{conf.prefix}-{conf.backbone_model_name}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    else:
        log_basename = f"{conf.prefix}-{conf.backbone_model_name}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    log_filename = f"{log_basename}.log"
    logzero.logfile(os.path.join(conf.log_dir, log_filename))

    # メインのnode以外のloggerを停止
    if hvd.rank() == 0:
        logzero.logfile(os.path.join(conf.log_dir, log_filename))
    else:
        logzero.logfile('', disableStderrLogger=True)

    logger.info(conf)

    OUTPUT_DIR = f"{conf.model_dir}/{conf.prefix}_{conf.backbone_model_name}"

    if hvd.rank() == 0:
        os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)
    logger.info(f"output_dir: {OUTPUT_DIR}")

    train = get_folds(conf, N_FOLDS, FOLD_SEED, DATA, logger)

    trn_fold = [int(elem) for elem in conf.train_fold.split(',')]

    total_epochs = conf.epochs * len(trn_fold)

    oof_y_true = []
    oof_y_preds = []
    oof_df = pd.DataFrame()
    submit_df_list = []
    for fold in range(conf.n_fold):
        if fold in trn_fold:
            best_score, best_y_true, best_y_preds, _oof_df =\
                train_loop(conf, hvd, train, fold, logger, log_basename, total_epochs,
                           args.new_train, args.new_test)
            if args.new_train or args.new_test:
                if args.ensemble_sigmoid:
                    _oof_df['preds'] = torch.tensor(_oof_df['preds'].values).sigmoid().numpy()
                submit_df_list.append(_oof_df)
                if args.new_test:
                    prefix = 'new_test'
                elif args.new_train:
                    prefix = 'new_train'
                elif args.test:
                    prefix = 'oof'
                else:
                    prefix = 'oof'
                _oof_df.to_csv(f"{OUTPUT_DIR}/{prefix}_fold{fold}.csv", index=False)
            else:
                oof_df = pd.concat([oof_df, _oof_df])
            oof_y_true.append(best_y_true)
            oof_y_preds.append(best_y_preds)
            logger.info(f"fold{fold} Best Score: {best_score:.4f}")
            total_epochs -= conf.epochs

    if args.new_train or args.new_test:
        sub_df = None
        if not args.new_test:
            for oof_df in submit_df_list:
                if sub_df is not None:
                    sub_df['preds'] = sub_df['preds'] + oof_df['preds']
                else:
                    sub_df = oof_df
            score = roc_auc_score(sub_df.target.values, sub_df.preds.values)
            logger.info(f"oof test score: {score}")
        else:
            for oof_df in submit_df_list:
                if sub_df is not None:
                    sub_df['target'] = sub_df['target'] + oof_df['preds']
                else:
                    oof_df = oof_df.drop('target', axis=1)
                    oof_df.columns = ['id', 'target']
                    sub_df = oof_df
        if hvd.rank() == 0:
            sub_df = sub_df.sort_values(by=['id']).reset_index(drop=True)
            if args.new_train:
                sub_df.to_csv(f"{OUTPUT_DIR}/new_train.csv", index=False)
            if args.new_test:
                sub_df.to_csv(f"{OUTPUT_DIR}/new_test.csv", index=False)
    else:
        if len(trn_fold) == N_FOLDS:
            oof_y_true = np.concatenate(oof_y_true)
            oof_y_preds = np.concatenate(oof_y_preds)
            score = roc_auc_score(oof_y_true, oof_y_preds)
            logger.info(f"oof score: {score}")
            if hvd.rank() == 0:
                oof_df = oof_df.sort_values(by=['id']).reset_index(drop=True)
                oof_df.to_csv(f"{OUTPUT_DIR}/oof_df.csv", index=False)

    logger.info(f"log saved: {os.path.join(conf.log_dir, log_filename)}")

