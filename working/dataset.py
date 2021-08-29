import typing as tp

import torch
import numpy as np
import albumentations as A
import pandas as pd

from albumentations.pytorch import ToTensorV2
from pathlib import Path
from scipy import ndimage as nd
from PIL import Image


FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]


class SetiSimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths: tp.Sequence[FilePath],
        labels: tp.Sequence[Label],
        ids: tp.Sequence[str],
        transform: A.Compose,
        target_only: bool = True,
        in_chans: int = 1,
        with_id: bool = False,
        seed: int = 71
    ):
        """Initialize"""
        self.paths = paths
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.target_only = target_only
        self.in_chans = in_chans
        self.with_id = with_id
        self.seed = seed

    def __len__(self):
        """Return num of cadence snippets"""
        return len(self.paths)

    def __getitem__(self, index: int):
        """Return transformed image and label for given index."""
        path, label, id = self.paths[index], self.labels[index], self.ids[index]
        img = self._read_cadence_array(path)
        img = self.transform(image=img)["image"]

        if self.with_id:
            return img, label, id
        else:
            return img, label

    def _read_cadence_array(self, path: Path):
        """Read cadence file and reshape"""
        if self.target_only:
            img = np.load(path)[[0, 2, 4]]  # shape: (3, 273, 256)
            img = np.vstack(img)  # shape: (819, 256)
        else:
            img = np.load(path)  # shape: (6, 273, 256)
            img = np.vstack(img)  # shape: (1638, 256)
        img = img.astype("f")[..., np.newaxis]  # shape: (819, 256, 1) or (1638, 256, 1)
        return img


# Transforms
from albumentations import (
    Compose,
    Resize, RandomResizedCrop, CenterCrop,
    HorizontalFlip, VerticalFlip,
    ShiftScaleRotate,
    GaussNoise,
)

def get_transforms(conf, mode='justresize'):
    if mode == 'aug_v1':
        transforms = Compose([
            Resize(height=conf.scale_height, width=conf.scale_width,
                   interpolation=conf.interpolation),
            HorizontalFlip(p=conf.HorizontalFlip_prob),
            VerticalFlip(p=conf.VerticalFlip_prob),
            ShiftScaleRotate(p=conf.ShiftScaleRotate_prob,
                             shift_limit=conf.shift_limit,
                             scale_limit=conf.scale_limit,
                             rotate_limit=conf.rotate_limit,
                             border_mode=conf.border_mode,
                             value=conf.value,
                             mask_value=conf.mask_value),
            RandomResizedCrop(p=conf.RandomResizedCrop_prob,
                              height=conf.input_height, width=conf.input_width,
                              scale=[conf.crop_scale_min, conf.crop_scale_max],
                              ratio=[conf.crop_ratio_min, conf.crop_ratio_max],
                              interpolation=conf.interpolation),
            ToTensorV2(),
        ])
    elif mode == 'aug_v2':
        transforms = Compose([
            ShiftScaleRotate(p=conf.ShiftScaleRotate_prob,
                             shift_limit=conf.shift_limit,
                             scale_limit=conf.scale_limit,
                             rotate_limit=conf.rotate_limit,
                             border_mode=conf.border_mode,
                             value=conf.value,
                             mask_value=conf.mask_value,
                             interpolation=conf.interpolation),
            RandomResizedCrop(p=conf.RandomResizedCrop_prob,
                              height=conf.input_height, width=conf.input_width,
                              scale=[conf.crop_scale_min, conf.crop_scale_max],
                              ratio=[conf.crop_ratio_min, conf.crop_ratio_max],
                              interpolation=conf.interpolation),
            HorizontalFlip(p=conf.HorizontalFlip_prob),
            VerticalFlip(p=conf.VerticalFlip_prob),
            GaussNoise(p=conf.GaussNoise_prob, var_limit=[0.01, 0.05]),
            ToTensorV2(),
        ])
    elif mode == 'flip_justresize': # flip & just resize
        transforms = Compose([
                Resize(height=conf.input_height, width=conf.input_width,
                       interpolation=conf.interpolation),
                HorizontalFlip(p=conf.HorizontalFlip_prob),
                VerticalFlip(p=conf.VerticalFlip_prob),
                ToTensorV2(),
            ])
    elif mode == 'justresize': # just resize
        transforms = Compose([
                Resize(height=conf.input_height, width=conf.input_width,
                       interpolation=conf.interpolation),
                ToTensorV2(),
            ])
    elif mode == 'centercrop':
        transforms = Compose([
                Resize(height=conf.scale_height, width=conf.scale_width,
                       interpolation=conf.interpolation),
                CenterCrop(height=conf.input_height, width=conf.input_width),
                ToTensorV2(),
            ])
    else:
        raise NotImplementedError("invalid mode")

    return transforms
