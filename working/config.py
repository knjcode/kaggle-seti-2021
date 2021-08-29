import yaml
from dataclasses import dataclass


@dataclass
class Config:
    prefix: str = 'test000'
    backbone_model_name: str = 'resnet18d'
    pretrained: bool = True  # timm向けのpretrainedフラグ
    pooling: str = 'avg' # avg / max / gem
    ckpt_path: str = ''
    backbone_ckpt_path: str = ''
    amp: bool = True
    in_chans: int = 1  # 1 or 2
    load_strict: bool = True

    train: bool = True

    # for horovod ddp
    sync_bn: bool = True

    # psedo labels
    pseudo_label: bool = False

    # loss
    loss_type: str = 'bce'  # bce / focal
    focal_loss_gamma: float = 2.0

    # dropout
    drop_rate: float = None

    # GeM
    gemp_p: float = 3.0
    gemp_learn_p: bool = False
    overwrite_gem_p: float = None

    # triplet attention
    triplet_attention: bool = False
    triplet_kernel_size: int = 7

    # act_layer
    act_layer: str = None # relu / swish or silu / mish / leaky_relu / prelu ...

    target_only: bool = True

    scale_height: int = 640
    scale_width: int = 640
    input_height: int = 640
    input_width: int = 640

    train_bs: int = 64
    valid_bs: int = 128

    n_fold: int = 5
    train_fold: str = '0,1,2,3,4'
    seed: int = 71
    fold_seed: int = 71

    optimizer: str = 'madgrad'
    lr: float = 1e-2
    min_lr: float = 1e-7
    weight_decay: float = 0.
    momentum: float = 0.9
    nesterov: bool = False

    epochs: int = 30
    T_0: int = 30
    T_mult: int = 1
    warmup_epochs: int = 1
    factor: float = 0.1
    patience: int = 4
    plateau_eps: float = 1e-8
    plateau_mode: str = 'max'
    scheduler: str = 'LinearWarmupCosineAnnealingLR'  # ReduceLROnPlateau / CosineAnnealingLR / CosineAnnealingWarmRestarts
                                                      # LinearWarmupCosineAnnealingLR / WarmupLinear

    train_trans_mode: str = 'aug_v1'
    valid_trans_mode: str = 'justresize'
    interpolation: int = 2  # 1:bilinear 2:bicubic 3:area 4:lanczos4

    # mixup
    mixup: bool = False
    mixup_alpha: float = 1.0

    RandomResizedCrop_prob: float = 0.5
    crop_scale_min: float = 0.9
    crop_scale_max: float = 1.0
    crop_ratio_min: float = 0.75
    crop_ratio_max: float = 1.3333333333333333

    HorizontalFlip_prob: float = 0.5
    VerticalFlip_prob: float = 0.5

    ShiftScaleRotate_prob: float = 0.2
    shift_limit: float = 0.2
    scale_limit: float = 0.2
    rotate_limit: int = 20
    border_mode: int = 4
    value: int = 0
    mask_value: int = 0

    GaussianNoise_prob: float = 0.2

    # tta
    tta_hflip: bool = False
    tta_vflip: bool = False
    tta_sigmoid: bool = False

    num_workers: int = 4
    device: str = 'cuda'
    pin_memory: bool = False
    model_dir: str = 'models'
    log_dir: str = 'logs'
    image_dump: bool = False
    print_freq: int = 0


def load_config(config_file):
    with open(config_file, 'r') as fp:
        opts = yaml.safe_load(fp)
    return Config(**opts)

