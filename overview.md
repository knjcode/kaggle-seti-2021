# 3rd place solution overview

[3rd place soluution (Triplet Attention)](https://www.kaggle.com/c/seti-breakthrough-listen/discussion/266403)


## Summary

Best submission was an emsemble of four EfficientNets with [Convolutional Triplet Attention Module](https://arxiv.org/abs/2010.03045).


## Validation and Preprocess

- Use new train data only
- StratifiedKFold(k=5)
- Use ON-channels only (819x256) and resize (768x768)


## Model Architecture

Backbone -> Triplet Attention -> GeM Pooling -> FC

- Use [EfficientNet](https://arxiv.org/abs/1905.11946) B1, B2, B3 and B4 backbone
- [Triplet Attention](https://arxiv.org/abs/2010.03045) (increase kernel size from 7 to 13)
  - Add only one attention layer after the backbone
  - I used the implementation in [here](https://github.com/rwightman/pytorch-image-models/blob/499790e117b2c8c1b57780b73d16c28b84db509e/timm/models/layers/triplet.py)
- [GeM Pooling](https://arxiv.org/abs/1711.02512) (p=4)
- To accelerate the training, replace Swish to ReLU (B3 and B4 only)


## Training

The training process consists of three stages

### Stage 0

This model will not be used for the final submission.

Training EfficientNet-B4

- 60epochs DDP AMP
- Focal Loss (gamma=0.5)
- [MADGRAD](https://github.com/facebookresearch/madgrad) optimizer
  - Initial Lr 1e-2 LinearWarmupCosineAnnealingLR (warmup_epochs=1)
- Mixup (alpha=1.0)
  - If the target of either data is 1, mixed_target will also be set to 1
- Data Augmentation
  - Horizontal and Vertical flip (p=0.5)
  - ShiftScaleRotate (shift_limit=0.2, scale_limit=0.2, rotate_limit=0)
  - RandomResizedCrop (scale=[0.8,1.0], ratio=[0.75,1.333])

Generate Pseudo labels from oof prediction of this model from new test set.  
(Select 5,000 images each of positive and negative data with high confidence)

### Stage 1

Training EfficientNet-B1,B2,B3 and B4 by adding pseudo labeled images.

The training settings are almost the same as stage0.

- 60epochs DDP AMP
- Focal Loss (gamma=0.5)
- [MADGRAD](https://github.com/facebookresearch/madgrad) optimizer
  - Initial Lr 1e-2 or 1e-3 LinearWarmupCosineAnnealingLR (warmup_epochs=5)
- Mixup (alpha=1.0)
  - If the target of either data is 1, mixed_target will also be set to 1
- Data Augmentation
  - Horizontal and Vertical flip (p=0.5)
  - ShiftScaleRotate (shift_limit=0.2, scale_limit=0.2, rotate_limit=0)
  - RandomResizedCrop (scale=[0.8,1.0], ratio=[0.75,1.333])


### Stage2

Refine stage1 models and generate submission with TTA

I refined the model with lighter data augmentation than stage1 for 10epochs.

- 10epochs DDP AMP
- Focal Loss (gamma=0.5)
- MADGRAD optimizer
  - Initial Lr 1e-4 LinearWarmupCosineAnnealingLR (warmup_epochs=0)
- Horizontal and Vertical flip only with Mixup
- 4xTTA in inference
  - regular, hflip, vflip, hflip+vflip


Best Single Model  (Efficientnet-B4)
- 5foldCV: 0.9048   PublicLB: 0.80586   PrivateLB: 0.80294

Ensemble of 4 EfficinetNets
- 5foldCV: 0.9100   PublicLB: 0.80575   PrivateLB: 0.80475


## Reference

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Rotate to Attend: Convolutional Triplet Attention Module](https://arxiv.org/abs/2010.03045)
- [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/abs/1711.02512)
