# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import pickle
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np

from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionMetrics, DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample

from tools.evaluation.detection_class import DetectionBox, EvalBoxes, DETECTION_NAMES
from tools.evaluation.algo import accumulate, calc_ap, calc_tp

EVAL_CONFIG = {
    "class_range": {
        "Car": 50,
    },
    "dist_fcn": "center_distance",
    "dist_ths": [0.5, 1.0, 2.0, 4.0],
    "dist_th_tp": 2.0,
    "min_recall": 0.1,
    "min_precision": 0.1,
    "max_boxes_per_sample": 500,
    "mean_ap_weight": 5
}

class DetectionConfig:
    """ Data class that specifies the detection evaluation settings. """

    def __init__(self,
                 class_range,
                 dist_fcn,
                 dist_ths,
                 dist_th_tp,
                 min_recall,
                 min_precision,
                 max_boxes_per_sample,
                 mean_ap_weight):
        
        assert set(class_range.keys()) == set(DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = self.class_range.keys()

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self):
        """ Serialize instance into json-friendly format. """
        return {
            'class_range': self.class_range,
            'dist_fcn': self.dist_fcn,
            'dist_ths': self.dist_ths,
            'dist_th_tp': self.dist_th_tp,
            'min_recall': self.min_recall,
            'min_precision': self.min_precision,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'mean_ap_weight': self.mean_ap_weight
        }

    @classmethod
    def deserialize(cls, content):
        """ Initialize from serialized dictionary. """
        return cls(content['class_range'],
                   content['dist_fcn'],
                   content['dist_ths'],
                   content['dist_th_tp'],
                   content['min_recall'],
                   content['min_precision'],
                   content['max_boxes_per_sample'],
                   content['mean_ap_weight'])

    @property
    def dist_fcn_callable(self):
        """ Return the distance function corresponding to the dist_fcn string. """
        if self.dist_fcn == 'center_distance':
            return center_distance
        else:
            raise Exception('Error: Unknown distance function %s!' % self.dist_fcn)

class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                #  nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 gt_path: str,
                #  eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        
        # with open(result_path, 'r') as f:
        #     pred = json.loads(result_path)["results"]
        # print(len(pred))
        
        # self.nusc = nusc
        # self.eval_set = eval_set
        self.result_path = result_path
        self.gt_path = gt_path
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = self.load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        self.gt_boxes = self.load_gt(gt_path, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), "Samples in split doesn't match samples in predictions."
        
        # Add center distances.
        self.pred_boxes = self.add_center_dist(self.pred_boxes)
        self.gt_boxes = self.add_center_dist(self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = self.filter_eval_boxes(self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = self.filter_eval_boxes(self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens
    
    def load_prediction(self, result_path, max_boxes_per_sample, box_cls, verbose=False):
        # Load from file and check that the format is correct.
        with open(result_path) as f:
            data = json.load(f)
        assert 'results' in data, 'Error: No field `results` in result file.'

        # Deserialize results and get meta data.
        all_results = EvalBoxes.deserialize(data['results'], box_cls)
        meta = data['meta']
        if verbose:
            print("Loaded results from {}. Found detections for {} samples.".format(result_path, len(all_results.sample_tokens)))

        # Check that each sample has no more than x predicted boxes.
        for sample_token in all_results.sample_tokens:
            assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
                "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

        return all_results, meta

    
    def load_gt(self, gt_path, box_cls, verbose=False):
        # Init.
        with open(gt_path, 'rb') as f:
            gt_all = pickle.load(f)
        if verbose:
            print('Loading annotations from {}'.format(gt_path))
    
        # Read out all sample_tokens in DB.
        assert len(gt_all) > 0, "Error: Database has no samples!"

        all_annotations = EvalBoxes()
        # Load annotations and filter predictions and annotations.
        for sample_token, gt in enumerate(gt_all):
            pred_boxes = gt["gt_boxes"]
            pred_names = gt["gt_names"]

            pred_velocity = None
            if 'gt_velocity' in gt:
                pred_velocity = gt["gt_velocity"]

            sample_boxes = []
            for box_idx in range(pred_boxes.shape[0]):
                if box_cls == DetectionBox:
                    # Get label name in detection task and filter unused labels.
                    bbox = pred_boxes[box_idx]
                    detection_name = pred_names[box_idx]
                    if pred_velocity is not None:
                        velocity = pred_velocity[box_idx]
                    if detection_name is None or detection_name not in DETECTION_NAMES:
                        continue

                    sample_boxes.append(
                        box_cls(
                            sample_token=str(sample_token),
                            translation=bbox[:3].tolist(),
                            size=bbox[3:6].tolist(),
                            rotation=bbox[6:].tolist(),
                            velocity=(0, 0) if pred_velocity is None else tuple(velocity[:2]),
                            detection_name=detection_name,
                            detection_score=-1.0,  # GT samples do not have a score.
                            attribute_name=""
                        )
                    )
                else:
                    raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

            all_annotations.add_boxes(str(sample_token), sample_boxes)

        if verbose:
            print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

        return all_annotations
    
    def add_center_dist(self, eval_boxes):
        for sample_token in eval_boxes.sample_tokens:
            for box in eval_boxes[sample_token]:
                # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
                # Note that the z component of the ego pose is 0.
                ego_translation = (box.translation[0] - box.ego_translation[0],
                                box.translation[1] - box.ego_translation[1],
                                box.translation[2] - box.ego_translation[2])
                if isinstance(box, DetectionBox):
                    box.ego_translation = ego_translation
                else:
                    raise NotImplementedError

        return eval_boxes
    
    def filter_eval_boxes(self, eval_boxes, max_dist, verbose):
        # Accumulators for number of filtered boxes.
        total, dist_filter, point_filter = 0, 0, 0
        for sample_token in eval_boxes.sample_tokens:
            # Filter on distance first.
            total += len(eval_boxes[sample_token])
            eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if box.ego_dist < max_dist[box.detection_name]]
            dist_filter += len(eval_boxes[sample_token])

            # # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
            # eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
            # point_filter += len(eval_boxes[sample_token])

        if verbose:
            print("=> Original number of boxes: %d" % total)
            print("=> After distance based filtering: %d" % dist_filter)
            # print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)

        return eval_boxes

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = False) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err']))

        return metrics_summary


if __name__ == "__main__":

    # Settings.
    # parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
    #                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    # parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
    #                     help='Folder to store result metrics, graphs and example visualizations.')
    # parser.add_argument('--eval_set', type=str, default='val',
    #                     help='Which dataset split to evaluate on, train, val or test.')
    # parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
    #                     help='Default nuScenes data directory.')
    # parser.add_argument('--version', type=str, default='v1.0-trainval',
    #                     help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    # parser.add_argument('--config_path', type=str, default='',
    #                     help='Path to the configuration file.'
    #                          'If no path given, the CVPR 2019 configuration will be used.')
    # parser.add_argument('--plot_examples', type=int, default=10,
    #                     help='How many example visualizations to write to disk.')
    # parser.add_argument('--render_curves', type=int, default=1,
    #                     help='Whether to render PR and TP curves to disk.')
    # parser.add_argument('--verbose', type=int, default=1,
    #                     help='Whether to print to stdout.')
    # args = parser.parse_args()

    # result_path_ = os.path.expanduser(args.result_path)
    # output_dir_ = os.path.expanduser(args.output_dir)
    # plot_examples_ = args.plot_examples
    # render_curves_ = bool(args.render_curves)
    # verbose_ = bool(args.verbose)


    #test
    cfg_ = DetectionConfig.deserialize(EVAL_CONFIG)

    result_path = "/root/code/work_dirs/1-26-nus_shtperson/pts_bbox/results_nusc.json"
    gt_path = "/remote-home/linmo2333/nuScenes_forward/ff_fullview_val_infos.pkl"
    output_dir = os.path.join(*os.path.split(result_path)[:-1])
    print(output_dir)

    detection_eval = DetectionEval(
            config=cfg_,
            result_path=result_path,
            gt_path=gt_path,
            output_dir=output_dir,
            verbose=True
    )
    detection_eval.main()
