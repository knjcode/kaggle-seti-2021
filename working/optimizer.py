from torch.optim import Adam, SGD
from timm.optim import RAdam
from madgrad import MADGRAD
from transformers.optimization import AdamW
from timm.optim.adabelief import AdaBelief
from timm.optim.adamp import AdamP
from timm.optim.sgdp import SGDP


def get_optimizer(conf, parameters):
    lr = float(conf.lr)
    wd = float(conf.weight_decay)

    if conf.optimizer == 'adam':
        optimizer = Adam(parameters, lr=lr, weight_decay=wd, amsgrad=False)
    elif conf.optimizer == 'amsgrad':
        optimizer = Adam(parameters, lr=lr, weight_decay=wd, amsgrad=True)
    elif conf.optimizer == 'adamw':
        optimizer = AdamW(parameters, lr=lr, weight_decay=wd)
    elif conf.optimizer == 'radam':
        optimizer = RAdam(parameters, lr=lr, weight_decay=wd)
    elif conf.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=lr, weight_decay=wd, momentum=conf.momentum)
    elif conf.optimizer == 'adabelief':
        optimizer = AdaBelief(parameters, lr=lr, weight_decay=wd, eps=1e-16,
                            weight_decouple=True, rectify=False, fixed_decay=False, amsgrad=False)
    elif conf.optimizer == 'madgrad':
        optimizer = MADGRAD(parameters, lr=lr, weight_decay=wd, momentum=conf.momentum)
    elif conf.optimizer == 'adamp':
        optimizer = AdamP(parameters, lr=lr, weight_decay=wd, nesterov=conf.nesterov)
    elif conf.optimizer == 'sgdp':
        optimizer = SGDP(parameters, lr=lr, weight_decay=wd, momentum=conf.momentum, nesterov=conf.nesterov)

    return optimizer
