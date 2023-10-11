# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import auto_fp16
from torch import nn

from ..builder import MIDDLE_ENCODERS

class BasicBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Initializes convolutional block for channel reduce
        Args:
            out_channels [int]: Number of output channels of convolutional block
            **kwargs [Dict]: Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                                bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features [torch.Tensor(B, C_in, H, W)]: Input features
        Returns:
            x [torch.Tensor(B, C_out, H, W)]: Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x

@MIDDLE_ENCODERS.register_module()
class VoxelToBEV(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        
        self.block = BasicBlock2D(in_channels=self.in_channels,
                                  out_channels=self.out_channels)

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features):
        """Foraward function to scatter features."""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases
        B, C, Z, Y, X = voxel_features.shape
        bev_features = voxel_features.view(B, C*Z, Y, X)  # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        bev_features = self.block(bev_features)  # (B, C*Z, Y, X) -> (B, C, Y, X)
        return bev_features

