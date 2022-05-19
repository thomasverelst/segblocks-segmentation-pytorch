from itertools import chain
from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import SpatialPyramidPooling, _BNReluConv, _Upsample
import segblocks

class SwiftNet(nn.Module):
    def __init__(self, backbone, num_classes=0,  *, num_features=128, k_up=3,
                 spp_grids=(8, 4, 2, 1), spp_square_grid=False, spp_drop_rate=0.0,
                 upsample_skip=True, upsample_only_skip=False,
                 output_stride=4, separable=False,
                 upsample_separable=False,
                 scale_factors=[2,2,2],
                  **kwargs):
        super(SwiftNet, self).__init__()
        self.backbone = backbone
        assert num_classes > 0
        self.num_classes = num_classes
        up_features = self.backbone.block_features
        self.num_features = num_features
        self.separable = separable

        self.fine_tune = [self.backbone]

        upsamples = []
        upsamples += [
            _Upsample(num_features, up_features[0], num_features, use_bn=True, k=k_up, use_skip=upsample_skip,
                      only_skip=upsample_only_skip, separable=upsample_separable, scale_factor=scale_factors[0])]
        upsamples += [
            _Upsample(num_features, up_features[1], num_features, use_bn=True, k=k_up, use_skip=upsample_skip,
                      only_skip=upsample_only_skip, separable=upsample_separable, scale_factor=scale_factors[1])]
        if scale_factors[2] > 1:
            upsamples += [
                _Upsample(num_features, up_features[2], num_features, use_bn=True, k=k_up, use_skip=upsample_skip,
                        only_skip=upsample_only_skip, separable=upsample_separable, scale_factor=scale_factors[2])]


        num_levels = 3
        self.spp_size = num_features
        bt_size = self.spp_size

        level_size = self.spp_size // num_levels
        self.spp = SpatialPyramidPooling(up_features[3 if scale_factors[2] > 1 else 2], num_levels, bt_size=bt_size, level_size=level_size,
                                        out_size=num_features, grids=spp_grids, square_grid=spp_square_grid,
                                        bn_momentum=0.01/2, use_bn=True, drop_rate=spp_drop_rate)
        num_up_remove = max(0, int(log2(output_stride) - 2))
        self.upsample = nn.ModuleList(list(reversed(upsamples[num_up_remove:])))

        self.logits = _BNReluConv(self.num_features, self.num_classes, batch_norm=True, k=1, bias=True)
        self.random_init = [self.spp, self.upsample, self.logits]

        for m in nn.ModuleList(self.random_init):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_down(self, image):
        features = self.backbone.forward_features(image)
        return features

    def forward_up(self, features):
        features = features[::-1] # reverse order
        x = features[0]
        x = segblocks.no_blocks(self.spp)(x)
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)

        # Final logits
        x = self.logits(x)
        return x

    def forward(self, image):
        return self.forward_up(self.forward_down(image))
