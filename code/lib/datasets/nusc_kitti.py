import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.utils import get_angle_from_box3d,check_range
from lib.datasets.kitti_utils import get_objects_from_label
from lib.datasets.kitti_utils import Calibration
from lib.datasets.kitti_utils import get_affine_transform
from lib.datasets.kitti_utils import affine_transform
from lib.datasets.kitti_utils import compute_box_3d
import pdb

class NUSC_KITTI(data.Dataset):
    def __init__(self, root_dir, split, cfg):
        # basic configuration
        self.num_classes = 10
        self.max_objs = 50
        self.class_name        = ["barrier"    , "bicycle"   , "bus"    , "car"    , "construction_vehicle"    , "motorcycle"    , "pedestrian"   , "traffic_cone"   , "trailer"    , "truck"]
        self.cls2id            = {"barrier": 0 , "bicycle": 1, "bus": 2 , "car": 3 , "construction_vehicle": 4 , "motorcycle": 5 , "pedestrian": 6, "traffic_cone": 7, "trailer": 8 , "truck": 9}
        self.cls_min_lidar_pts = {"barrier": 10, "bicycle": 5, "bus": 40, "car": 35, "construction_vehicle": 20, "motorcycle": 10, "pedestrian": 5, "traffic_cone": 4, "trailer": 40, "truck": 30}
        # Max distance threshold used by nuscenes evaluation
        self.cls_max_distance  = {"barrier": 30, "bicycle":40, 'bus': 50, "car": 50, "construction_vehicle": 50, "motorcycle": 40, "pedestrian":40, "traffic_cone":30, "trailer": 50, "truck": 50}
        self.resolution = np.array([672, 384])  if 'resolution' not in cfg.keys() else np.array(cfg['resolution'])
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])

        self.box2d_orig_hmin = 20 if 'box2d_orig_hmin' not in cfg.keys() else cfg['box2d_orig_hmin']

        ##w,h,l
        self.cls_mean_size = np.array([[0.9856928278, 2.5350272733, 0.4964593291],
                                       [1.3212777838, 0.6161487411, 1.7345529633],
                                       [3.5310034111, 2.9322881261, 11.2886576022],
                                       [1.730323945 , 1.9504693996,  4.6300396845],
                                       [3.2509390814, 2.8786857152,  7.0579567879],
                                       [1.5086667691, 0.7789657637,  2.1047384053],
                                       [1.7830991759, 0.6741489141,  0.7322679154],
                                       [1.0864875408, 0.4343957855,  0.4397158911],
                                       [3.8569623772, 2.8859517869, 12.2184129639],
                                       [2.9017936138, 2.530615423 ,  7.0539758051]])

        # data split loading
        assert split in ['train', 'val', 'train_small', 'val_small', 'train_5k', 'train_10k', 'trainval', 'test'] or 'train' in split
        self.split = split
        split_dir = os.path.join(root_dir, 'nusc_kitti', 'ImageSets', split + '.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # path configuration
        self.data_dir = os.path.join(root_dir, 'nusc_kitti', 'validation' if 'val' in split else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label')

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'train_small', 'trainval'] else False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)    # (H, W, 3) RGB mode


    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)


    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        img_size = np.array(img.size)

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False
        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_crop:
                random_crop_flag = True
                crop_size = img_size * np.clip(np.random.randn()*self.scale + 1, 1 - self.scale, 1 + self.scale)
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        coord_range = np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32)
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        calib = self.get_calib(index)
        features_size = self.resolution // self.downsample# W * H
        #  ============================   get labels   ==============================
        if self.split!='test':
            objects = self.get_label(index)
            # data augmentation for labels
            if random_flip_flag:
                calib.flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.ry = np.pi - object.ry
                    object.pos[0] *= -1
                    if object.ry > np.pi:  object.ry -= 2 * np.pi
                    if object.ry < -np.pi: object.ry += 2 * np.pi
            # labels encoding
            heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32) # C * H * W
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            height2d = np.zeros((self.max_objs, 1), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
            mask_3d = np.zeros((self.max_objs), dtype=np.uint8)
            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
            for i in range(object_num):
                # filter objects by writelist
                if objects[i].cls_type not in self.writelist:
                    continue

                # filter inappropriate samples by difficulty
                if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                    continue

                # filter objects by max_distance
                if objects[i].pos[-1] > self.cls_max_distance[objects[i].cls_type]:
                    continue

                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()

                # filter boxes with zero or negative 2D height or width
                temp_w, temp_h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                if temp_w <= 0 or temp_h <= 0:
                    continue

                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                # modify the 2d bbox according to pre-compute downsample ratio
                bbox_2d[:] /= self.downsample

                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
                center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                center_3d = affine_transform(center_3d.reshape(-1), trans)
                center_3d /= self.downsample

                # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
                if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
                if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue

                # generate the radius of gaussian heatmap
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))

                if objects[i].cls_type in ['Void', 'DontCare', 'void', 'ignore']:
                    draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                    continue

                cls_id = self.cls2id[objects[i].cls_type]
                cls_ids[i] = cls_id
                draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)

                # encoding 2d/3d offset & 2d size
                indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                offset_2d[i] = center_2d - center_heatmap
                size_2d[i] = 1. * w, 1. * h

                # encoding depth
                depth[i] = objects[i].pos[-1]

                # encoding heading angle
                #heading_angle = objects[i].alpha
                heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0]+objects[i].box2d[2])/2)
                if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi: heading_angle += 2 * np.pi
                heading_bin[i], heading_res[i] = angle2class(heading_angle)

                # encoding 3d offset & size_3d
                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                size_3d[i] = src_size_3d[i] - mean_size

                # nuScenes does NOT have truncation or occlusion labels
                # filter objects by num_lidar_points_per_box.
                # num_lidar_points_per_box are a proxy for occlusion.
                # lower num_lidar_points_ber_box means more occlusion.
                # Note that the 15th column is treated as score in Object3d of kitti_utils.py but is num_lidar_points_ber_box
                num_lidar_points_per_box = objects[i].score
                if num_lidar_points_per_box <= self.cls_min_lidar_pts[objects[i].cls_type]:
                    continue

                #objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
                box2d_orig_h = objects[i].box2d[3]-objects[i].box2d[1]
                #if box2d_orig_h >= self.box2d_orig_hmin:
                mask_2d[i] = 1
            targets = {'depth': depth,
                   'size_2d': size_2d,
                   'heatmap': heatmap,
                   'offset_2d': offset_2d,
                   'indices': indices,
                   'size_3d': size_3d,
                   'offset_3d': offset_3d,
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'cls_ids': cls_ids,
                   'mask_2d': mask_2d}
        else:
            targets = {}
        # collect return data
        inputs = img
        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size}
        return inputs, calib.P2, coord_range, targets, info   #calib.P2
