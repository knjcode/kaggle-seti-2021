import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import get_act_layer

from config import Config
from triplet_attention import TripletAttention


class GeMP(nn.Module):
    def __init__(self, p=3., eps=1e-6, learn_p=False):
        super().__init__()
        self._p = p
        self._learn_p = learn_p
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.set_learn_p(flag=learn_p)

    def set_learn_p(self, flag):
        self._learn_p = flag
        self.p.requires_grad = flag

    def forward(self, x):
        x = F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

        return x


class BlockAttentionModel(nn.Module):
    def __init__(
        self,
        conf: Config,
        backbone: nn.Module,
        n_features: int,
    ):
        """Initialize"""
        super(BlockAttentionModel, self).__init__()
        self.conf = conf
        self.backbone = backbone
        self.n_features = n_features
        self.drop_rate = conf.drop_rate
        self.pooling = conf.pooling
        if conf.act_layer is not None:
            act_layer = get_act_layer(conf.act_layer)
        else:
            act_layer = nn.ReLU

        if conf.triplet_attention:
            self.attention = TripletAttention(self.n_features,
                                              act_layer=act_layer,
                                              kernel_size=conf.triplet_kernel_size)
        else:
            self.attention = nn.Identity()

        if self.pooling == 'avg':
            self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        elif self.pooling == 'gem':
            self.global_pool = GeMP(p=conf.gemp_p, learn_p=conf.gemp_learn_p)
        elif self.pooling == 'max':
            self.global_pool = torch.nn.AdaptiveMaxPool2d(1)
        elif self.pooling == 'nop':
            self.global_pool = torch.nn.Identity()
        else:
            raise NotImplementedError(f'Invalid pooling type: {self.pooling}')

        self.head = nn.Linear(self.n_features, 1)


    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        if type(self.fc.bias) == torch.nn.parameter.Parameter:
            nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


    def forward(self, x, t=None):
        """Forward"""
        x = self.backbone(x)
        x = self.attention(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.head(x)
        return x


def get_model(conf, backbone_name, logger):
    # create backbone

    if backbone_name in ['tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns',
                         'tf_efficientnet_b2_ns', 'tf_efficientnet_b3_ns',
                         'tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ns',
                         'tf_efficientnet_b6_ns', 'tf_efficientnet_b7_ns']:
        kwargs = {}
        if conf.act_layer:
            act_layer = get_act_layer(conf.act_layer)
            kwargs['act_layer'] = act_layer

        backbone = timm.create_model(backbone_name, pretrained=True,
                                    in_chans=conf.in_chans, **kwargs)
        n_features = backbone.num_features
        backbone.reset_classifier(0, '')

    else:
        raise NotImplementedError(f'not implemented yet: {backbone_name}')

    if conf.backbone_ckpt_path:
        ckpt = torch.load(conf.backbone_ckpt_path, map_location='cpu')
        check = backbone.load_state_dict(ckpt, strict=conf.load_strict)
        logger.info(check)
        logger.info(f"load backbone weight from: {conf.backbone_ckpt_path}")

    model = BlockAttentionModel(conf, backbone, n_features)

    if conf.ckpt_path:
        ckpt = torch.load(conf.ckpt_path, map_location='cpu')
        check = model.load_state_dict(ckpt['model'], strict=conf.load_strict)
        logger.info(f"load model weight from : {conf.ckpt_path}")
        logger.info(f"check: {check}")

    return model

