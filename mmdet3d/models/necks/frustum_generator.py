# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import math
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS

import time
class metric:
    def __init__(self, n=0, avg=0):
        self.n = n
        self.avg = avg
    
    def add(self, num):
        self.avg = (self.n*self.avg + num) / (self.n + 1)
        self.n = self.n + 1

@NECKS.register_module()
class FrustumGenerator(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 disc_cfg,
                 init_cfg=None):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(FrustumGenerator, self).__init__(init_cfg=init_cfg)
        self.disc_cfg = disc_cfg

        # self.up = nn.Sequential(
        #     nn.ConvTranspose2d(
        #             in_channels=2048,
        #             out_channels=1024,
        #             kernel_size=4,
        #             stride=2,
        #             padding=1,
        #             output_padding=0,
        #             bias=False),
        #     nn.BatchNorm2d(1024, momentum=0.1),
        #     nn.ReLU(inplace=True)
        # )

        # self.reduce = nn.Sequential(
        #     nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(512, momentum=0.1),
        #     nn.ReLU(inplace=True)
        # )

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        self.metric_dict = {
            "reduce": metric(),
            "select":metric(),
            "onehot":metric(),
            "create": metric()
        }
        
        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    def bin_depths(self, depth_map, target=False):
        """
        Converts depth map into bin indices
        Args:
            depth_map [torch.Tensor(H, W)]: Depth Map
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            indices [torch.Tensor(H, W)]: Depth bin indices
        """
        mode = self.disc_cfg['mode']
        depth_max = self.disc_cfg['depth_max']
        depth_min = self.disc_cfg['depth_min']
        num_bins = self.disc_cfg['num_bins']

        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        elif mode == "SID":
            indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
                (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError
        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)
        return indices
    
    def create_frustum_features(self, image_features, depth_onehot):
        """
        Create image depth feature volume by multiplying image features with depth classification scores
        Args:
            image_features [torch.Tensor(N, C, H, W)]: Image features
            depth_onehot [torch.Tensor(N, D, H, W)]: Depth classification in one-hot form
        Returns:
            frustum_features [torch.Tensor(N, C, D, H, W)]: Image features
        """
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)
        depth_onehot = depth_onehot.unsqueeze(channel_dim)

        # Multiply to form image depth feature volume
        frustum_features = depth_onehot * image_features
        return frustum_features

    @auto_fp16()
    def forward(self, img_feats, img_depth):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        start = time.time()
        if isinstance(img_feats, tuple):
            # img_feat = torch.cat([self.up(img_feats[-1]), img_feats[-2]], dim=1) 
            img_feat = self.reduce(img_feat)
        else:
            img_feat = img_feats
            img_feat = self.reduce(img_feat)
        
        reduce_end = time.time()
        self.metric_dict['reduce'].add(reduce_end - start)

        img_depth_disc = self.bin_depths(img_depth, target=True)
        b, h, w = list(img_depth_disc.shape)
        img_depth_disc = img_depth_disc.view(-1)
        img_depth_onehot = torch.eye(self.disc_cfg['num_bins'] + 1, device=img_depth_disc.device)
        img_depth_onehot = img_depth_onehot.index_select(0, img_depth_disc)

        # img_depth_onehot = torch.zeros((b*h*w, self.disc_cfg['num_bins'] + 1), device=img_depth_disc.device)
        # img_depth_onehot[:, img_depth_disc] = 1
        
        select_end = time.time()
        self.metric_dict['select'].add(select_end - reduce_end)

        img_depth_onehot = img_depth_onehot.view(b, h, w, -1)
        img_depth_onehot = img_depth_onehot.permute(0, 3, 1, 2).contiguous() 
        img_depth_onehot = img_depth_onehot[:, :self.disc_cfg['num_bins'], :, :]

        onehot_end = time.time()
        self.metric_dict['onehot'].add(onehot_end - select_end)

        frustum_features = self.create_frustum_features(img_feat, img_depth_onehot)

        # B, C, H, W = img_feat.shape
        # D = self.disc_cfg['num_bins']
        # frustum_features_new = torch.zeros((B, C, D + 1, H, W), dtype=img_feat.dtype, device=img_feat.device)
        # frustum_features_new = frustum_features_new.permute(0, 3, 4, 2, 1).contiguous().view(-1, D + 1, C)
        # img_depth_disc = img_depth_disc.view(-1)
        # img_feat = img_feat.permute(0, 2, 3, 1).contiguous().view(-1, C)
        # frustum_features_new[torch.arange(B*H*W), img_depth_disc] = img_feat 
        # frustum_features_new = frustum_features_new.view(B, H, W, D + 1, C)
        # frustum_features_new = frustum_features_new.permute(0, 4, 3, 1, 2).contiguous()
        # frustum_features_new = frustum_features_new[:, :, :self.disc_cfg['num_bins'], :, :]

        create_end = time.time()
        self.metric_dict['create'].add(create_end - onehot_end)
        
        # s = ""
        # for k, v in self.metric_dict.items():
        #     s = s + "{}: {:.4f}s | ".format(k, v.avg)
        # print(s)

        return frustum_features
