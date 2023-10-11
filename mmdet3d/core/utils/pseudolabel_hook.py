import os
import json
import pickle

import torch
from torch.utils.data import DataLoader

from mmcv.runner import HOOKS, Hook
from mmcv.parallel import collate

import random
import copy
import numpy as np
from functools import partial

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

@HOOKS.register_module()
class pseudolabel_hook(Hook):

    def __init__(self,
                 gt_path,
                 test_cfg,
                 train_cfg, 
                 prototype_root,
                 prototype_file,
                 interval):
        from mmdet3d.datasets import build_dataset

        test_cfg.ann_file = gt_path
        train_cfg.data_root = prototype_root
        train_cfg.ann_file = prototype_file

        self.gt_path = gt_path
        self.prototype_file = prototype_file
        self.interval = interval

        self.dataset = build_dataset(test_cfg)
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=1,
            collate_fn=partial(collate, samples_per_gpu=1),
            pin_memory=False)
        
        self.prototype_dataset = build_dataset(train_cfg)
        self.prototype_data_loader = DataLoader(
            self.prototype_dataset,
            batch_size=1,
            num_workers=1,
            collate_fn=partial(collate, samples_per_gpu=1),
            pin_memory=False)
    
    def center_box_to_corners(self, box):
        pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, yaw = box
        half_dim_x, half_dim_y, half_dim_z = dim_x/2.0, dim_y/2.0, dim_z/2.0
        corners = np.array([[half_dim_x, half_dim_y, -half_dim_z],
                            [half_dim_x, -half_dim_y, -half_dim_z],
                            [-half_dim_x, -half_dim_y, -half_dim_z],
                            [-half_dim_x, half_dim_y, -half_dim_z],
                            [half_dim_x, half_dim_y, half_dim_z],
                            [half_dim_x, -half_dim_y, half_dim_z],
                            [-half_dim_x, -half_dim_y, half_dim_z],
                            [-half_dim_x, half_dim_y, half_dim_z]])
        transform_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0, pos_x],
            [np.sin(yaw), np.cos(yaw), 0, pos_y],
            [0, 0, 1.0, pos_z],
            [0, 0, 0, 1.0],
        ])
        corners = (transform_matrix[:3, :3] @ corners.T + transform_matrix[:3, [3]]).T
        return corners
    
    def bboxes_filter_by_points(self, bboxes_cornors, points, threshold=1):
        idx_chosen = []
        point_chosen = []
        for idx in range(bboxes_cornors.shape[0]):
            cornors = bboxes_cornors[idx]

            p1 = cornors[2, :]
            p_x = cornors[1, :]
            p_y = cornors[3, :]
            p_z = cornors[6, :]

            i = p_x - p1
            j = p_y - p1
            k = p_z - p1

            v = points.T - p1.reshape((-1, 1))

            iv = np.dot(i.reshape((1, -1)), v)
            jv = np.dot(j.reshape((1, -1)), v)
            kv = np.dot(k.reshape((1, -1)), v)

            mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
            mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
            mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
            mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

            if np.sum(mask) >= threshold:
                idx_chosen.append(idx)
                point_chosen.append(copy.deepcopy(points[mask.squeeze()]))
        return idx_chosen, point_chosen
    
    def in_range_3d(self, points, point_range):
        in_range_flags = ((points[:, 0] > point_range[0])
                          & (points[:, 1] > point_range[1])
                          & (points[:, 2] > point_range[2])
                          & (points[:, 0] < point_range[3])
                          & (points[:, 1] < point_range[4])
                          & (points[:, 2] < point_range[5]))
        return in_range_flags
    
    def get_prototype(self, runner):
        runner.model.eval()
        for _, data in enumerate(self.prototype_data_loader):
            with torch.no_grad():
                runner.model(return_loss=True, **data)
        
        runner.model.module.init_prototypes()
        runner.model.module.set_prototype_mode(False)

    def before_run(self, runner):
        # if not runner.model.module.get_prototype_mode():
        #     runner.logger.info('No Need to Generate Prototype')
        # else:
        #     runner.logger.info('Generate Prototype from {}'.format(self.prototype_data_loader.dataset.ann_file))
        #     self.get_prototype(runner)
        pass

    def before_train_epoch(self, runner):
        from mmdet3d.apis import single_gpu_test

        CLASSES_NAME = ['Car']
        scores_threshold = [0.2]
        point_density = [1]

        # if runner.epoch % self.interval == 0 and runner.epoch != 0:
        if runner.epoch % self.interval == 0:
            runner.logger.info('Generate Pseudo Label from {}'.format(self.gt_path))
            results = single_gpu_test(runner.model, self.data_loader, show=False)
            result_files, _ = self.data_loader.dataset.format_results(results)

            with open(result_files["pts_bbox"], 'r') as f:
                preds = json.load(f)["results"]

            with open(self.gt_path, 'rb') as f:
                gts = pickle.load(f)
            
            assert len(gts) == len(preds), ('The length of predictions is not equal to the dataset len: {} != {}'.format(len(preds), len(gts)))

            save_dir, save_name = os.path.split(self.gt_path)

            infos = []
            for idx, key in enumerate(preds):
                gts_it = gts[idx]
                lidar_path = gts[idx]['lidar_path']
                # pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 3)
                # point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
                # mask = self.in_range_3d(pc, point_cloud_range)
                # pc = pc[mask]

                pred_boxes = preds[key]["pred_boxes"]
                pred_names = preds[key]["pred_names"]
                pred_score = preds[key]["pred_scores"]
                pred_velocity = None

                if "pred_velocity" in preds[key]:
                    pred_velocity = preds[key]["pred_velocity"]
                    pred_velocity = np.array(pred_velocity, dtype=np.float32)

                pred_score = np.array(pred_score, dtype=np.float32)
                pred_boxes = np.array(pred_boxes, dtype=np.float32)
                pred_names = np.array(pred_names)

                #new method
                boxes_cls_list, names_cls_list, velocity_cls_list = [], [], []
                for i, cls in enumerate(CLASSES_NAME):
                    class_ind = pred_names == cls
                    if np.sum(class_ind) == 0:
                        continue
                    names_cls = pred_names[class_ind] 
                    boxes_cls = pred_boxes[class_ind] 
                    velocity_cls = pred_velocity[class_ind] if pred_velocity is not None else None
                    scores_cls = pred_score[class_ind] 

                    # filte by score
                    score_ind = scores_cls > scores_threshold[i]
                    if np.sum(score_ind) == 0:
                        continue
                    names_cls = names_cls[score_ind] 
                    boxes_cls = boxes_cls[score_ind] 
                    velocity_cls = velocity_cls[score_ind] if velocity_cls is not None else None
                    scores_cls = scores_cls[score_ind] 

                    # # filte by point density
                    # cornors_cls = [self.center_box_to_corners(boxes_cls[i]) for i in range(boxes_cls.shape[0])]
                    # cornors_cls = np.concatenate(cornors_cls, axis=0).reshape(-1, 8, 3)
                    # idx_chosen, points_chose = self.bboxes_filter_by_points(cornors_cls, pc, threshold=point_density[i])
                    # boxes_cls = boxes_cls[idx_chosen]
                    # names_cls = names_cls[idx_chosen]
                    # scores_cls = scores_cls[idx_chosen]

                    boxes_cls_list.append(boxes_cls)
                    names_cls_list.append(names_cls)
                    velocity_cls_list.append(velocity_cls)
                
                if len(boxes_cls_list) == 1:
                    pred_boxes = boxes_cls_list[0]
                    pred_names = names_cls_list[0]
                    pred_velocity = velocity_cls_list[0] if pred_velocity is not None else None
                elif len(boxes_cls_list) > 1:
                    pred_boxes = np.concatenate(boxes_cls_list, axis=0)
                    pred_names = np.concatenate(names_cls_list, axis=0)
                    pred_velocity = np.concatenate(velocity_cls_list, axis=0) if pred_velocity is not None else None

                # #old method
                # mask = pred_score > 0.2
                # pred_boxes = pred_boxes[mask]
                # pred_names = pred_names[mask]

                # info = {
                #     'lidar_path': lidar_path,
                #     'gt_boxes': pred_boxes,
                #     'gt_names': pred_names,
                # }

                gts_it['gt_boxes'] = pred_boxes
                gts_it['gt_names'] = pred_names
                if pred_velocity is not None:
                    gts_it['gt_velocity'] = pred_velocity

                infos.append(gts_it)
            
            info_path = os.path.join(save_dir, 'tmp_pseudo_' + save_name)
            with open(info_path, 'wb') as f:
                pickle.dump(infos, f)
            runner.logger.info('Pseudo label save at {}...'.format(info_path))

            runner.data_loader.dataset.data_infos = runner.data_loader.dataset.load_annotations(info_path)