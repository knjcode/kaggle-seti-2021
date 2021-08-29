from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup
)


class ReduceLROnPlateauPatch(ReduceLROnPlateau):
    def get_last_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]


def get_scheduler(conf, optimizer, train_loader, logger):
    min_lr = float(conf.min_lr)
    factor = float(conf.factor)
    eps = float(conf.plateau_eps)
    n_iter_per_epoch = len(train_loader)

    if conf.scheduler=='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateauPatch(optimizer, mode=conf.plateau_mode,
                                           factor=factor, patience=conf.patience,
                                           verbose=True, eps=eps)
        logger.info(f"use ReduceLROnPlateau Scheduler factor:{factor} patience:{conf.patience}")
    elif conf.scheduler=='CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=conf.epochs * n_iter_per_epoch,
                                      eta_min=min_lr, last_epoch=-1)
        logger.info(f"use CosineAnnealingLR Scheduler epochs:{conf.epochs}")
    elif conf.scheduler=='CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=conf.T_0 * n_iter_per_epoch, T_mult=conf.T_mult, eta_min=min_lr)
    elif conf.scheduler=='LinearWarmupCosineAnnealingWarmRestarts':
        if not conf.epochs == 1:
            assert (conf.epochs - conf.warmup_epochs) % conf.T_0 == 0
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=conf.warmup_epochs * n_iter_per_epoch,
            num_training_steps=conf.epochs * n_iter_per_epoch,
            num_cycles=(conf.epochs - conf.warmup_epochs) // conf.T_0
        )
    elif conf.scheduler=='LinearWarmupCosineAnnealingLR':
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                  num_warmup_steps=conf.warmup_epochs * n_iter_per_epoch,
                                  num_training_steps=conf.epochs * n_iter_per_epoch)
    elif conf.scheduler=='WarmupLinear':
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=conf.warmup_epochs * n_iter_per_epoch)

    return scheduler

