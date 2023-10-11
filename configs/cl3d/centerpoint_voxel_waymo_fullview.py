_base_ = [
    'model_centerpoint_voxel.py',
    '../_base_/schedules/cyclic_20e.py', 
    '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
eval_range = 50

class_names = [
    'Car'
]

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(num_features=3),
    pts_middle_encoder=dict(in_channels=3),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

dataset_type = 'PandasetDataset'
data_root = '/remote-home/linmo2333/waymo_infos/'
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=3,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadPointsFromPrevSweeps',
    #     load_dim=3,
    #     use_dim=4,
    #     file_client_args=file_client_args,
    #     use_pn=False,
    #     remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=3,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadPointsFromPrevSweeps',
    #     load_dim=3,
    #     use_dim=4,
    #     file_client_args=file_client_args,
    #     use_pn=False,
    #     remove_close=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=3,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadPointsFromPrevSweeps',
    #     load_dim=3,
    #     use_dim=4,
    #     file_client_args=file_client_args,
    #     use_pn=False,
    #     remove_close=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'ff_train_infos_new.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        eval_range=eval_range,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'ff_val_infos_new.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        eval_range=eval_range,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'ff_val_infos_new.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        eval_range=eval_range,
        box_type_3d='LiDAR')
)

evaluation = dict(interval=20, pipeline=eval_pipeline)
runner = dict(type='EpochBasedRunner', max_epochs=20)


# # prototype_root = '/remote-home/linmo2333/nuScenes_forward/'
# # prototype_file = prototype_root + 'ff_fullview_train_infos.pkl'
# prototype_root = data_root
# prototype_file = data_root + 'test0.2_pseudo_ff_patchnorm_train_infos.pkl'

# custom_hooks = [
#     dict(
#         type='pseudolabel_hook', 
#         gt_path=data_root + 'withvelo_patchnorm_train_infos.pkl', 
#         test_cfg=None, 
#         train_cfg=None, 
#         prototype_root=prototype_root,
#         prototype_file=prototype_file,
#         interval=1,
#         priority='NORMAL')
# ]


work_dir = './work_dirs/7-4-waymo'