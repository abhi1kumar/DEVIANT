"""
    Sample Run:
     /home/abc/anaconda3/envs/py36_waymo_tf/bin/python -u data/waymo/waymo_eval.py --predictions output/model/result_test/data --pd_set data/waymo/ImageSets/val.txt
     /home/abc/anaconda3/envs/py36_waymo_tf/bin/python -u data/waymo/waymo_eval.py --sanity # To run sanity check, APs should be 1.00

    Waymo evaluation on kitti style outputs
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
import argparse
import pdb
import pandas
import numpy as np
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2

ERROR = 1e-6

import pandas as pd
import warnings

def read_csv(path, delimiter= " ", ignore_warnings= False, use_pandas= False):
    try:
        if ignore_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if use_pandas:
                    data = pd.read_csv(path, delimiter= delimiter, header=None).values
                else:
                    data = np.genfromtxt(path, delimiter= delimiter)
        else:
            if use_pandas:
                data = pd.read_csv(path, delimiter=delimiter, header=None).values
            else:
                data = np.genfromtxt(path, delimiter=delimiter)
    except:
        data = None

    return data


class DetectionMetricsEstimatorTest(tf.test.TestCase):

    def get_boxes_from_bin(self, file):
        pd_bbox, pd_type, pd_frame_id, pd_score, difficulty = [], [], [], [], []
        stuff1 = metrics_pb2.Objects()
        with open(file, 'rb')as rf:
            stuff1.ParseFromString(rf.read())
            for i in range(len(stuff1.objects)):
                obj = stuff1.objects[i].object
                pd_frame_id.append(stuff1.objects[i].frame_timestamp_micros)
                box = [obj.box.center_x, obj.box.center_y, obj.box.center_z,
                       obj.box.length, obj.box.width, obj.box.height, obj.box.heading]
                pd_bbox.append(box)
                pd_score.append(stuff1.objects[i].score)
                pd_type.append(obj.type)

                if obj.num_lidar_points_in_box and obj.num_lidar_points_in_box<=5:
                    difficulty.append(2)
                else:
                    difficulty.append(1)
        return np.array(pd_bbox), np.array(pd_type), np.array(pd_frame_id), np.array(pd_score), np.array(difficulty)

    def get_boxes_from_txt(self, pd_set, gt_set, pd_dir, gt_dir):
        __type_list = {'unknown': 0, 'Car': 1, 'Pedestrian': 2, 'Sign': 3, 'Cyclist': 4}
        pd_bbox, pd_type, pd_frame_id, pd_score, difficulty = [], [], [], [], []
        gt_bbox, gt_type, gt_frame_id, gt_score, gt_diff    = [], [], [], [], []

        f = open(gt_set, 'r')
        gt_lines = f.readlines()
        f.close()

        f = open(pd_set, 'r')
        pred_lines = f.readlines()
        f.close()

        for i in range(len(pred_lines)): #39848
            file_name     = pred_lines[i].strip() + '.txt'
            pred_file     = os.path.join(pd_dir, file_name)

            # 000123 can be on the first line of val.txt file, but it refers to the 123rd line of val_org.txt
            gt_map_id     = int(pred_lines[i])
            gt_seg, gt_id = gt_lines[gt_map_id].strip().split(' ')
            gt_file_name  = os.path.join(gt_dir , 'validation_org', gt_seg, 'label_0', gt_id + '.txt')

            if not os.path.exists(pred_file) or not os.path.exists(gt_file_name):
                continue

            # Read predictions
            pred_data  = read_csv(path= pred_file, delimiter= " ", ignore_warnings= False, use_pandas= True)
            if pred_data is not None:
                pred_class = pred_data[:, 0].tolist()

                pred_other = pred_data.copy()
                pred_other[:, 0] = 0
                pred_other = pred_other.astype(float)
                num_pred   = pred_data.shape[0]

                if num_pred > 0:
                    pd_frame_id += [gt_id] * num_pred
                    pd_bbox     += pred_other[:, [11, 12, 13, 10, 9, 8, 14]].tolist()
                    pd_score    += pred_other[:, 15].tolist()
                    pd_type     += [__type_list[x] for x in pred_class]
                    difficulty  += [1] * num_pred

            # Read ground truth
            gt_data   = read_csv(path= gt_file_name, delimiter= " ", ignore_warnings= False, use_pandas= True)
            if gt_data is not None:
                gt_class  = gt_data[:, 0]

                gt_other  = gt_data.copy()
                gt_other[:, 0] = 0
                gt_other  = gt_other.astype(float)

                #   0  1   2    3      4   5   6  7    8    9   10   11   12    13   14   15
                # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, lidar
                gt_bad_index  = np.logical_or(
                                    np.logical_or(gt_other[:, 15] == 0,
                                                  gt_other[:, 4]-gt_other[:, 6] >=0),
                                    gt_other[:, 5]-gt_other[:, 7] >=0)
                gt_good_index = np.logical_not(gt_bad_index)

                num_gt   = np.sum(gt_good_index)

                if num_gt > 0:
                    gt_class = gt_class[gt_good_index].tolist()
                    gt_other = gt_other[gt_good_index]

                    gt_diff_temp = gt_other[:, 15]
                    level_1_index = gt_diff_temp > 5
                    level_2_index = gt_diff_temp <= 5
                    gt_diff_temp[level_2_index] = 2
                    gt_diff_temp[level_1_index] = 1

                    gt_frame_id += [gt_id] * num_gt
                    gt_bbox     += gt_other[:, [11, 12, 13, 10, 9, 8, 14]].tolist()
                    gt_score    += ['0.5'] * num_gt
                    gt_type     += [__type_list[x] for x in gt_class]
                    gt_diff     += gt_diff_temp.tolist()

            if (i+1)% 1000 ==0 or (i+1) == len(pred_lines) :
                print('{} images read'.format(i+1))

        return np.array(pd_bbox, dtype=np.float32), \
               np.array(pd_type, dtype=np.uint8), \
               np.array(pd_frame_id, dtype=np.int64), \
               np.array(pd_score, dtype=np.float32), \
               np.array(difficulty), \
               np.array(gt_bbox, dtype=np.float32), \
               np.array(gt_type, dtype=np.uint8), \
               np.array(gt_frame_id, dtype=np.int64), \
               np.array(gt_score, dtype=np.float32), \
               np.array(gt_diff, dtype=np.uint8)

    def _BuildConfig(self):
        config = metrics_pb2.Config()
        # pdb.set_trace()
        config_text = """
    num_desired_score_cutoffs: 11
    breakdown_generator_ids: OBJECT_TYPE
    breakdown_generator_ids: RANGE
    difficulties {
    levels: 1
    levels: 2
    }
    difficulties {
    levels: 1
    levels: 2
    }
    matcher_type: TYPE_HUNGARIAN
    iou_thresholds: 0.0
    iou_thresholds: 0.7
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    box_type: TYPE_3D
    """
        text_format.Merge(config_text, config)
        return config

    def _BuildGraph(self, graph):
        with graph.as_default():
            self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._gt_difficulty = tf.compat.v1.placeholder(dtype=tf.uint8)
            metrics = detection_metrics.get_detection_metric_ops(
                config=self._BuildConfig(),
                prediction_frame_id=self._pd_frame_id,
                prediction_bbox=self._pd_bbox,
                prediction_type=self._pd_type,
                prediction_score=self._pd_score,
                prediction_overlap_nlz=tf.zeros_like(
                    self._pd_frame_id, dtype=tf.bool),
                ground_truth_bbox=self._gt_bbox,
                ground_truth_type=self._gt_type,
                ground_truth_frame_id=self._gt_frame_id,
                ground_truth_difficulty=self._gt_difficulty,
                recall_at_precision=0.95,
            )
            return metrics

    def _EvalUpdateOps(
            self,
            sess,
            graph,
            metrics,
            prediction_frame_id,
            prediction_bbox,
            prediction_type,
            prediction_score,
            ground_truth_frame_id,
            ground_truth_bbox,
            ground_truth_type,
            ground_truth_difficulty,
    ):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])],
            feed_dict={
                self._pd_bbox: prediction_bbox,
                self._pd_frame_id: prediction_frame_id,
                self._pd_type: prediction_type,
                self._pd_score: prediction_score,
                self._gt_bbox: ground_truth_bbox,
                self._gt_type: ground_truth_type,
                self._gt_frame_id: ground_truth_frame_id,
                self._gt_difficulty: ground_truth_difficulty,
            })

    def _EvalValueOps(self, sess, graph, metrics):
        ddd = {}
        for item in metrics.items():
            ddd[item[0]] = sess.run([item[1][0]])
        return ddd

    def testAPBasic(self):
        print("start")
        print("=> Pred_folder: ", pd_dir)
        print("=> Pred_set:    ", pd_set)
        print("=> GT_set:      ", gt_set)

        pd_bbox, pd_type, pd_frame_id, pd_score, _, gt_bbox, gt_type, gt_frame_id, gt_score, difficulty = \
            self.get_boxes_from_txt(pd_set, gt_set, pd_dir, gt_dir)

        graph = tf.Graph()
        metrics = self._BuildGraph(graph)
        with self.test_session(graph=graph) as sess:
            sess.run(tf.compat.v1.initializers.local_variables())
            if FLAGS.sanity:
                # Pass on gt stuff as predictions
                self._EvalUpdateOps(sess, graph, metrics, gt_frame_id, gt_bbox, gt_type,
                                    gt_score, gt_frame_id, gt_bbox, gt_type, difficulty)
            else:
                self._EvalUpdateOps(sess, graph, metrics, pd_frame_id, pd_bbox, pd_type,
                                    pd_score, gt_frame_id, gt_bbox, gt_type, difficulty)

            aps = self._EvalValueOps(sess, graph, metrics)
            category_list  = ["VEHICLE", "CYCLIST", "PEDESTRIAN", "SIGN"]
            level_list  = [1, 2]
            metric_list = ["AP", "APH", "Recall@0.95"]
            print("--------------------------------------------------------------------------------------------")
            print("Class      | L |         {:11s}     |         {:11s}     |     {:11s}     ".format("AP_3D", "APH_3D", "Recall@0.95"))
            print("--------------------------------------------------------------------------------------------")
            for category in category_list:
                for level in level_list:
                    text = "{:10s} | {} ".format(category, level)
                    key_list = ["OBJECT_TYPE_TYPE_{}_LEVEL_{}".format(category, level), \
                                "RANGE_TYPE_{}_[0, 30)_LEVEL_{}".format(category, level),\
                                "RANGE_TYPE_{}_[30, 50)_LEVEL_{}".format(category, level),\
                                "RANGE_TYPE_{}_[50, +inf)_LEVEL_{}".format(category, level)]
                    for metric in metric_list:
                        key0 = os.path.join(key_list[0], metric)
                        key1 = os.path.join(key_list[1], metric)
                        key2 = os.path.join(key_list[2], metric)
                        key3 = os.path.join(key_list[3], metric)
                        # Report in percentage.
                        multiplier = 100.0
                        text += "| {:5.2f} {:5.2f} {:5.2f} {:5.2f} ".format(multiplier * aps[key0][0],
                                                                            multiplier * aps[key1][0],
                                                                            multiplier * aps[key2][0],
                                                                            multiplier * aps[key3][0])
                    print(text)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('predictions','','Directory to put the training data')
flags.DEFINE_string('pd_set',     '','Prediction validation set ')
flags.DEFINE_bool('sanity', False, 'Sanity Check Script')

if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    folder_path = os.path.join(os.getcwd(), 'data/waymo')

    # pd_set and gt_set are the validation set index, pd_set is converted to the index pattern from gt_set using set_split.py
    if FLAGS.pd_set == "":
        pd_set = os.path.join(folder_path, 'ImageSets/val_small.txt')
    else:
        pd_set = os.path.join(os.getcwd(), FLAGS.pd_set)
    gt_set = os.path.join(folder_path, 'ImageSets/val_org.txt')

    # pd_dir is path of the predicted data for val set,
    if FLAGS.predictions == "":
        pd_dir = os.path.join(folder_path, 'validation/label')
    else:
        pd_dir = os.path.join(os.getcwd(), FLAGS.predictions)
    # gt_dir is the folder in which validation_org folder (generated waymo label in kitti format) is located.
    gt_dir = folder_path #os.path.join(folder_path, 'validation/label' )

    tf.compat.v1.disable_eager_execution()
    tf.test.main()
