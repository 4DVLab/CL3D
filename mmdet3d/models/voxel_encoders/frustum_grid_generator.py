import torch
import torch.nn as nn
import kornia
import math

class FrustumGridGenerator(nn.Module):

    def __init__(self, voxel_size, pc_range, disc_cfg):
        """
        Initializes Grid Generator for frustum features
        Args:
            grid_size [np.array(3)]: Voxel grid shape [X, Y, Z]
            pc_range [list]: Voxelization point cloud range [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
            disc_cfg [int]: Depth discretiziation configuration
        """
        super().__init__()
        self.dtype = torch.float32
        self.voxel_size = torch.as_tensor(voxel_size)
        self.pc_range = pc_range
        self.out_of_bounds_val = -2
        self.disc_cfg = disc_cfg

        # Calculate voxel size
        pc_range = torch.as_tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.grid_size = (self.pc_max - self.pc_min) / self.voxel_size

        # Create voxel grid
        self.depth, self.width, self.height = self.grid_size.int()
        self.voxel_grid = kornia.utils.create_meshgrid3d(depth=self.depth,
                                                         height=self.height,
                                                         width=self.width,
                                                         normalized_coordinates=False)

        self.voxel_grid = self.voxel_grid.permute(0, 1, 3, 2, 4)  # XZY-> XYZ

        # Add offsets to center of voxel
        self.voxel_grid += 0.5
        self.grid_to_lidar = self.grid_to_lidar_unproject(pc_min=self.pc_min,
                                                          voxel_size=self.voxel_size)
                                                
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

    def project_to_image(self, project, points):
        """
        Project points to image
        Args:
            project [torch.tensor(..., 3, 4)]: Projection matrix
            points [torch.Tensor(..., 3)]: 3D points
        Returns:
            points_img [torch.Tensor(..., 2)]: Points in image
            points_depth [torch.Tensor(...)]: Depth of each point
        """
        # Reshape tensors to expected shape
        points = kornia.convert_points_to_homogeneous(points)
        points = points.unsqueeze(dim=-1)
        project = project.unsqueeze(dim=1)

        # Transform points to image and get depths
        points_t = project @ points
        points_t = points_t.squeeze(dim=-1)
        points_img = kornia.convert_points_from_homogeneous(points_t)
        points_depth = points_t[..., -1] - project[..., 2, 3]

        return points_img, points_depth
    
    def normalize_coords(self, coords, shape):
        """
        Normalize coordinates of a grid between [-1, 1]
        Args:
            coords [torch.Tensor(..., 2)]: Coordinates in grid
            shape [torch.Tensor(2)]: Grid shape [H, W]
        Returns:
            norm_coords [torch.Tensor(.., 2)]: Normalized coordinates in grid
        """
        min_n, max_n = -1, 1
        D, H, W = shape[:3]

        mask = coords[:, :, :, :, 0] > 0
        mask = torch.logical_and(mask, coords[:, :, :, :, 0] < W - 1)
        mask = torch.logical_and(mask, coords[:, :, :, :, 1] > 0)
        mask = torch.logical_and(mask, coords[:, :, :, :, 1] < H - 1)
        mask = torch.logical_and(mask, coords[:, :, :, :, 2] > 0)
        mask = torch.logical_and(mask, coords[:, :, :, :, 2] < D - 1)
        mask = torch.logical_not(mask)

        # Subtract 1 since pixel indexing from [0, shape - 1]
        norm_shape = torch.tensor([W, H, D], dtype=coords.dtype, device=coords.device)
        norm_coords = coords / (norm_shape - 1) * (max_n - min_n) + min_n
        norm_coords[mask] = self.out_of_bounds_val
        return norm_coords, mask

    def grid_to_lidar_unproject(self, pc_min, voxel_size):
        """
        Calculate grid to LiDAR unprojection for each plane
        Args:
            pc_min [torch.Tensor(3)]: Minimum of point cloud range [X, Y, Z] (m)
            voxel_size [torch.Tensor(3)]: Size of each voxel [X, Y, Z] (m)
        Returns:
            unproject [torch.Tensor(4, 4)]: Voxel grid to LiDAR unprojection matrix
        """
        x_size, y_size, z_size = voxel_size
        x_min, y_min, z_min = pc_min
        unproject = torch.tensor([[x_size, 0, 0, x_min],
                                  [0, y_size, 0, y_min],
                                  [0,  0, z_size, z_min],
                                  [0,  0, 0, 1]],
                                 dtype=self.dtype)  # (4, 4)

        return unproject

    # def transform_grid(self, voxel_grid, grid_to_lidar, lidar_to_cam, cam_to_img):
    #     """
    #     Transforms voxel sampling grid into frustum sampling grid
    #     Args:
    #         grid [torch.Tensor(B, X, Y, Z, 3)]: Voxel sampling grid
    #         grid_to_lidar [torch.Tensor(4, 4)]: Voxel grid to LiDAR unprojection matrix
    #         lidar_to_cam [torch.Tensor(B, 4, 4)]: LiDAR to camera frame transformation
    #         cam_to_img [torch.Tensor(B, 3, 4)]: Camera projection matrix
    #     Returns:
    #         frustum_grid [torch.Tensor(B, X, Y, Z, 3)]: Frustum sampling grid
    #     """
    #     B = lidar_to_cam.shape[0]

    #     # Create transformation matricies
    #     V_G = grid_to_lidar  # Voxel Grid -> LiDAR (4, 4)
    #     C_V = lidar_to_cam.type_as(grid_to_lidar)  # LiDAR -> Camera (B, 4, 4)
    #     I_C = cam_to_img.type_as(grid_to_lidar)  # Camera -> Image (B, 3, 4)
    #     trans = C_V @ V_G

    #     # Reshape to match dimensions
    #     trans = trans.reshape(B, 1, 1, 4, 4)
    #     voxel_grid = voxel_grid.repeat_interleave(repeats=B, dim=0)

    #     # Transform to camera frame
    #     camera_grid = kornia.transform_points(trans_01=trans, points_1=voxel_grid)

    #     # Project to image
    #     I_C = I_C.reshape(B, 1, 1, 3, 4)
    #     image_grid, image_depths = self.project_to_image(project=I_C, points=camera_grid)

    #     # Convert depths to depth bins
    #     image_depths = self.bin_depths(depth_map=image_depths)

    #     # Stack to form frustum grid
    #     image_depths = image_depths.unsqueeze(-1)
    #     frustum_grid = torch.cat((image_grid, image_depths), dim=-1)
        
    #     return frustum_grid

    def transform_grid(self, voxel_grid, grid_to_lidar, lidar_to_img):
        """
        Transforms voxel sampling grid into frustum sampling grid
        Args:
            grid [torch.Tensor(B, X, Y, Z, 3)]: Voxel sampling grid
            grid_to_lidar [torch.Tensor(4, 4)]: Voxel grid to LiDAR unprojection matrix
            lidar_to_cam [torch.Tensor(B, 4, 4)]: LiDAR to camera frame transformation
            cam_to_img [torch.Tensor(B, 3, 4)]: Camera projection matrix
        Returns:
            frustum_grid [torch.Tensor(B, X, Y, Z, 3)]: Frustum sampling grid
        """
        # print("transform_grid")
        B, N = lidar_to_img.shape[:2]

        # Create transformation matricies
        V_G = grid_to_lidar  # Voxel Grid -> LiDAR (4, 4)
        C_V = lidar_to_img.type_as(grid_to_lidar)  # LiDAR -> Image (B, 4, 4)
        trans = C_V @ V_G
        # print(trans.shape, voxel_grid.shape)

        # Reshape to match dimensions
        trans = trans.view(B*N, 4, 4)
        voxel_grid = voxel_grid.repeat_interleave(repeats=B*N, dim=0)
        _, W, H, D, C =  voxel_grid.shape
        voxel_grid = voxel_grid.view(B*N, -1, 3)
        # print(trans.shape, voxel_grid.shape)

        # project lidar grid to image plane
        voxel_grid = torch.cat([voxel_grid, torch.ones((*voxel_grid.shape[:2], 1))], dim=2)
        image_grid = trans[:, :3, :] @ voxel_grid.permute(0, 2, 1)
        image_grid = image_grid.permute(0, 2, 1).view(-1, W, H, D, C)
        image_depths = image_grid[:, :, :, :, 2]
        image_grid = image_grid[:, :, :, :, :2] / image_grid[:, :, :, :, 2:3]
        # print(image_grid.shape, image_depths.shape)

        # Convert depths to depth bins
        image_depths = self.bin_depths(depth_map=image_depths)
        # print(image_grid.shape, image_depths.shape)

        # Stack to form frustum grid
        image_depths = image_depths.unsqueeze(-1)
        frustum_grid = torch.cat((image_grid, image_depths), dim=-1)
        # print(frustum_grid.shape)
        
        return frustum_grid

    def forward(self, lidar_to_cam, cam_to_img, lidar_to_img, image_shape):
        """
        Generates sampling grid for frustum features
        Args:
            lidar_to_cam [torch.Tensor(B, 4, 4)]: LiDAR to camera frame transformation
            cam_to_img [torch.Tensor(B, 3, 4)]: Camera projection matrix
            image_shape [torch.Tensor(B, 2)]: Image shape [H, W]
        Returns:
            frustum_grid [torch.Tensor(B, X, Y, Z, 3)]: Sampling grids for frustum features
        """
        # frustum_grid = self.transform_grid(voxel_grid=self.voxel_grid.to(lidar_to_cam.device),
        #                                    grid_to_lidar=self.grid_to_lidar.to(lidar_to_cam.device),
        #                                    lidar_to_cam=lidar_to_cam,
        #                                    cam_to_img=cam_to_img)

        B, N = lidar_to_img.shape[:2]
        frustum_grid = self.transform_grid(voxel_grid=self.voxel_grid.to(lidar_to_img.device),
                                           grid_to_lidar=self.grid_to_lidar.to(lidar_to_img.device),
                                           lidar_to_img=lidar_to_img)

        # Normalize grid
        image_shape, _ = torch.max(image_shape, dim=0)
        image_depth = torch.tensor([self.disc_cfg["num_bins"]], device=image_shape.device, dtype=image_shape.dtype)
        frustum_shape = torch.cat((image_depth, image_shape))
        frustum_grid, mask = self.normalize_coords(coords=frustum_grid, shape=frustum_shape)

        # Replace any NaNs or infinites with out of bounds
        mask_nan = ~torch.isfinite(frustum_grid)
        frustum_grid[mask_nan] = self.out_of_bounds_val
        mask_nan = torch.any(mask_nan, dim=4)
        mask = torch.logical_or(mask, mask_nan)

        return frustum_grid, mask