# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.runner import BaseModule


@POSITIONAL_ENCODING.register_module()
class SinePositionalEncoding3D(nn.Module):
    def __init__(self,
                 num_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.):
        super().__init__()
        self.num_feats = num_feats
        self.num_pos_feats = num_feats
        self.temperature = temperature
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, x, mask=None):
        # b, t, c, h, w
        assert x.dim(
        ) == 5, f'{x.shape} should be a 5-dimensional Tensor, got {x.dim()}-dimensional Tensor instead'
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(1), x.size(3), x.size(4)),
                               device=x.device,
                               dtype=torch.bool)
        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            z_embed = (z_embed + self.offset) / (z_embed[:, -1:, :, :] +
                                                 self.eps) * self.scale
            y_embed = (y_embed + self.offset) / (y_embed[:, :, -1:, :] +
                                                 self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, :, -1:] +
                                                 self.eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32,
                             device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        dim_t_z = torch.arange((self.num_pos_feats * 2),
                               dtype=torch.float32,
                               device=x.device)
        dim_t_z = self.temperature**(2 * (dim_t_z // 2) /
                                     (self.num_pos_feats * 2))

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t_z
        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()),
            dim=5).flatten(4)
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()),
            dim=5).flatten(4)
        pos_z = torch.stack(
            (pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()),
            dim=5).flatten(4)
        pos = (torch.cat(
            (pos_y, pos_x), dim=4) + pos_z).permute(0, 1, 4, 2,
                                                    3)  # b, t, c, h, w
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str
