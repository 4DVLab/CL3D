# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import DynamicScatter
from .. import builder
from ..builder import VOXEL_ENCODERS
from .utils import VFELayer, get_paddings_indicator

from .frustum_grid_generator import FrustumGridGenerator


@VOXEL_ENCODERS.register_module()
class FrustumToVoxel(nn.Module):
    def __init__(self, sample_cfg, voxel_size, pc_range, disc_cfg):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            model_cfg [EasyDict]: Module configuration
            grid_size [np.array(3)]: Voxel grid shape [X, Y, Z]
            pc_range [list]: Voxelization point cloud range [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
            disc_cfg [dict]: Depth discretiziation configuration
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.disc_cfg = disc_cfg
        self.sample_cfg = sample_cfg
        self.grid_generator = FrustumGridGenerator(voxel_size=voxel_size, pc_range=pc_range, disc_cfg=disc_cfg)

    def forward(self, frustum_features, img_meta):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            batch_dict:
                frustum_features [torch.Tensor(B, C, D, H_image, W_image)]: Image frustum features
                lidar_to_cam [torch.Tensor(B, 4, 4)]: LiDAR to camera frame transformation
                cam_to_img [torch.Tensor(B, 3, 4)]: Camera projection matrix
                image_shape [torch.Tensor(B, 2)]: Image shape [H, W]
        Returns:
            batch_dict:
                voxel_features [torch.Tensor(B, C, Z, Y, X)]: Image voxel features
        """

        trans_lidar_to_cam = []
        trans_lidar_to_img = []
        trans_cam_to_img = []
        image_shape = []
        for img_meta_it in img_meta:
            trans_lidar_to_cam.append(torch.tensor(img_meta_it['lidar2cam'], dtype=torch.double))
            trans_lidar_to_img.append(torch.tensor(img_meta_it['lidar2img'], dtype=torch.double))
            trans_cam_to_img.append(torch.tensor(img_meta_it['cam2img'], dtype=torch.double))
            image_shape.append(torch.tensor(img_meta_it['img_shape']))
        trans_lidar_to_cam = torch.stack(trans_lidar_to_cam, dim=0)
        trans_lidar_to_img = torch.stack(trans_lidar_to_img, dim=0)
        trans_cam_to_img = torch.stack(trans_cam_to_img, dim=0)
        image_shape = torch.stack(image_shape, dim=0)
        # print(trans_lidar_to_cam.shape, trans_lidar_to_img.shape, trans_cam_to_img.shape)
        # print(image_shape)

        # Generate sampling grid for frustum volume
        grid, mask = self.grid_generator(lidar_to_cam=trans_lidar_to_cam,
                                   cam_to_img=trans_cam_to_img,
                                   lidar_to_img=trans_lidar_to_img,
                                   image_shape=image_shape)  # (B, X, Y, Z, 3)

        # Sample frustum volume to generate voxel volume
        grid = grid.to(frustum_features.device)
        mask = mask.to(frustum_features.device)

        B, N, C, W, H, D = frustum_features.shape
        voxel_features = torch.nn.functional.grid_sample(
            input=frustum_features.view(B*N, C, W, H, D), grid=grid, mode=self.sample_cfg['mode'], 
            padding_mode=self.sample_cfg['padding_mode'], align_corners=True
        )# (B, C, X, Y, Z)

        # (B, C, X, Y, Z) -> (B, C, Z, Y, X)
        # print(voxel_features.shape)

        mask = mask.unsqueeze(1).repeat(1, C, 1, 1, 1)
        voxel_features[mask] = 0
        voxel_features = voxel_features.view(B, N, *voxel_features.shape[1:])
        voxel_features = torch.sum(voxel_features, dim=1)
        voxel_features = voxel_features.permute(0, 1, 4, 3, 2)
        # print(voxel_features.shape)

        # test = torch.sum(voxel_features, dim=1)
        # print(torch.sum(test == 0))
        # exit(0)
        return voxel_features