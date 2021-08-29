from math import log
import time

import numpy as np
import torch
import pandas as pd

from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, LambdaLR

from util import AverageMeter, timeSince

try:
    from mpi4py import MPI
except:
    pass


def disable_bn(model):
  for module in model.modules():
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
      module.eval()

def enable_bn(model):
  model.train()


def train_fn(conf, global_step, train_loader, model, criterion, optimizer,
             epoch, scheduler, device, train_sampler, logger, tb_writer):

    if conf.device == 'cuda':
        scaler = GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    # train_samplerがNoneでなければhorovod ddp
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    start = end = time.time()
    for step, batch in enumerate(train_loader):
        global_step += 1

        optimizer.zero_grad()

        images, labels = batch

        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        with autocast(enabled=conf.amp):
            logits = model(images)
            loss = criterion(logits, labels)

        losses.update(loss.item(), batch_size)

        # backward
        if conf.amp:
            if train_sampler is not None:
                scaler.scale(loss).backward()
                optimizer.synchronize()
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
                scaler.update()
                optimizer.synchronize()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            if conf.sam:
                loss.backward()
                optimizer.first_step(zero_grad=True)
                disable_bn(model)
                criterion(model(images), labels).backward()
                optimizer.second_step(zero_grad=True)
                enable_bn(model)
            else:
                loss.backward()
                optimizer.step()

        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        elif isinstance(scheduler, LambdaLR):
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % conf.print_freq == 0 or step == (len(train_loader)-1):
            logger.info(f'Epoch: [{epoch+1}][{step+1}/{len(train_loader)}] '
                f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                f'Elapsed {timeSince(start, float(step+1)/len(train_loader)):s} '
                f'Loss: {losses.val:.4f}({losses.avg:.4f}) ')
            tb_writer.add_scalar('Loss/train', losses.avg, global_step)

            current_lr = scheduler.get_last_lr()
            tb_writer.add_scalar('Other/LearningRate', current_lr[0], global_step)

    return losses.avg, global_step


def mixup_train_fn(conf, global_step, train_loader, train_loader2, model, criterion, optimizer,
                   epoch, scheduler, device, train_sampler, train_sampler2, logger, tb_writer):

    if conf.device == 'cuda':
        scaler = GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    # train_samplerがNoneでなければhorovod ddp
    if train_sampler is not None:
        # 異なるsamplingになるようにset_epochの値を変える
        train_sampler.set_epoch(epoch)
        train_sampler2.set_epoch(epoch+10000)


    start = end = time.time()
    for step, (batch1, batch2) in enumerate(zip(train_loader, train_loader2)):
        global_step += 1

        optimizer.zero_grad()

        images1, labels1 = batch1
        images2, labels2 = batch2

        # measure data loading time
        data_time.update(time.time() - end)

        labels1 = labels1.to(device, non_blocking=True)
        labels2 = labels2.to(device, non_blocking=True)
        batch_size = labels1.size(0)

        alpha = conf.mixup_alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        mixed_images = lam * images1 + (1 - lam) * images2
        mixed_images = mixed_images.to(device, non_blocking=True)

        new_label = torch.clip(labels1 + labels2, 0, 1)

        with autocast(enabled=conf.amp):
            logits = model(mixed_images, new_label)
            loss = criterion(logits, new_label)

        losses.update(loss.item(), batch_size)

        # backward
        if conf.amp:
            if train_sampler is not None:
                scaler.scale(loss).backward()
                optimizer.synchronize()
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
                scaler.update()
                optimizer.synchronize()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            optimizer.step()


        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        elif isinstance(scheduler, LambdaLR):
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % conf.print_freq == 0 or step == (len(train_loader)-1):
            logger.info(f'Epoch: [{epoch+1}][{step+1}/{len(train_loader)}] '
                f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                f'Elapsed {timeSince(start, float(step+1)/len(train_loader)):s} '
                f'Loss: {losses.val:.4f}({losses.avg:.4f}) ')
            tb_writer.add_scalar('Loss/train', losses.avg, global_step)

            current_lr = scheduler.get_last_lr()
            tb_writer.add_scalar('Other/LearningRate', current_lr[0], global_step)

    return losses.avg, global_step


def valid_fn(conf, global_step, valid_loader, model, criterion, device, gather_results, hvd, logger, tb_writer, new_test):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    all_logits = []
    all_labels = []
    all_ids = []
    score = 0.5
    start = end = time.time()
    for step, batch in enumerate(valid_loader):
        images, labels, ids = batch

        all_labels.append(labels)
        all_ids.append(ids)

        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        with torch.no_grad():

            logits = model(images)

            # TTA
            if conf.tta_sigmoid and (conf.tta_hflip or conf.tta_vflip):
                logits = logits.sigmoid()

            if conf.tta_hflip:
                tta_logits = model(torch.flip(images, [3]))
                if conf.tta_sigmoid:
                    logits += tta_logits.sigmoid()
                else:
                    logits += tta_logits
            if conf.tta_vflip:
                tta_logits = model(torch.flip(images, [2]))
                if conf.tta_sigmoid:
                    logits += tta_logits.sigmoid()
                else:
                    logits += tta_logits
            if conf.tta_hflip and conf.tta_vflip:
                tta_logits = model(torch.flip(images, [2,3]))
                if conf.tta_sigmoid:
                    logits += tta_logits.sigmoid()
                else:
                    logits += tta_logits
                logits /= 4.
            elif conf.tta_hflip or conf.tta_vflip:
                logits /= 2.

            loss = criterion(logits, labels)

            losses.update(loss.item(), batch_size)

        all_logits.append(logits.detach().cpu().numpy().astype("f"))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % conf.print_freq == 0 or step == (len(valid_loader)-1):
            logger.info(f'EVAL: [{step+1}/{len(valid_loader)}] '
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        f'Elapsed {timeSince(start, float(step+1)/len(valid_loader)):s} ')

    tb_writer.add_scalar('Loss/valid', losses.avg, global_step)

    t = np.concatenate(all_labels)
    y = np.concatenate(all_logits)
    ids = np.concatenate(all_ids)

    if gather_results:  # for horovod
        comm = MPI.COMM_WORLD

        full_t = np.concatenate(comm.allgather(t))
        full_y = np.concatenate(comm.allgather(y))
        full_ids = np.concatenate(comm.allgather(ids))

        df_t = pd.DataFrame(full_t)
        df_y = pd.DataFrame(full_y)
        df_ids = pd.DataFrame(full_ids, columns=['id'])
        df_ids = df_ids.drop_duplicates(subset=['id']).sort_values(by=['id'])
        df_t = df_t.iloc[df_ids.index]
        df_y = df_y.iloc[df_ids.index]
        t = df_t.values
        y = df_y.values
        if not new_test:
            score = roc_auc_score(t, y)
            logger.info(f"AUROC score: {score}")
            tb_writer.add_scalar('AUROC', score, global_step)
    else:
        if not new_test:
            score = roc_auc_score(t, y)
            logger.info(f"AUROC score: {score}")
            tb_writer.add_scalar('AUROC', score, global_step)

    return losses.avg, score, t, y
