"""
    Sample Run:
    python plot/plot_qualitative_output.py --folder output/prevolution_nuscenes_3/results_test/data --dataset nuscenes
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import ImageFont

np.set_printoptions   (precision= 4, suppress= True)
torch.set_printoptions(precision= 4)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

import cv2
import imutils
import random

from lib.helpers.file_io import imread, imwrite
from lib.datasets.kitti_utils import get_calib_from_file, get_objects_from_label, convertRot2Alpha
from lib.helpers.util import *
# from lib.math_3d import *
# from lib.util import create_colorbar, draw_bev, draw_tick_marks, imhstack, draw_3d_box

import glob
import argparse

def plot_boxes_on_image_and_in_bev(predictions_img, plot_color, box_class_list= ["car", "cyclist", "pedestrian"], use_classwise_color= False, show_3d= True, show_bev= True, thickness= 4):
    # https://sashamaps.net/docs/resources/20-colors/
    # Some taken from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py#L10
    class_color_map = {'car': (255,51,153),
                       'cyclist': (255, 130, 48),  # Orange
                       'bicycle': (255, 130, 48),  # Orange
                       'pedestrian': (138, 43, 226),  # Violet
                       'bus': (0, 0, 0), # Black
                       'construction_vehicle': (0, 130, 200), # Blue
                       'motorcycle': (220, 190, 255),  # Lavender
                       'trailer': (170, 255, 195), # Mint
                       'truck': (128, 128, 99),  # Olive
                       'traffic_cone': (255, 225, 25), # Yellow
                       'barrier': (128, 128, 128),  # Grey
                       }

    if predictions_img is not None and predictions_img.size > 0:
        # Add dimension if there is a single point
        if predictions_img.ndim == 1:
            predictions_img = predictions_img[np.newaxis, :]

        N = predictions_img.shape[0]
        #   0   1    2     3   4   5  6    7    8    9    10   11   12   13    14      15
        # (cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score/num_lidar_points  )
        # Add projected 3d center information to predictions_img
        cls = predictions_img[:, 0]
        x1  = predictions_img[:, 4]
        y1  = predictions_img[:, 5]
        h3d = predictions_img[:, 8]
        w3d = predictions_img[:, 9]
        l3d = predictions_img[:, 10]
        x3d = predictions_img[:, 11]
        y3d = predictions_img[:, 12] - h3d/2
        z3d = predictions_img[:, 13]
        ry3d = predictions_img[:,14]

        if predictions_img.shape[1] > 15:
            score = predictions_img[:, 15]
        else:
            score = np.ones((predictions_img.shape[0],))

        for j in range(N):
            box_class = cls[j].lower()
            if box_class == "dontcare":
                continue
            if dataset == "nuscenes" or box_class in box_class_list:
                if use_classwise_color:
                    box_plot_color = class_color_map[box_class]
                else:
                    box_plot_color = plot_color

                # if box_class == "car" and score[j] < 100:
                #     continue
                # if box_class != "car" and score[j] < 50:
                #     continue

                box_plot_color = box_plot_color[::-1]
                if show_3d:
                    verts_cur, corners_3d_cur = project_3d(p2, x3d[j], y3d[j], z3d[j], w3d[j], h3d[j], l3d[j], ry3d[j], return_3d=True)
                    draw_3d_box(img, verts_cur, color= box_plot_color, thickness= thickness)
                    # cv2.putText(img, str(int(score[j])), (int(x1[j]), int(y1[j])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
                if show_bev:
                    draw_bev(canvas_bev, z3d[j], l3d[j], w3d[j], x3d[j], ry3d[j], color= box_plot_color, scale= bev_scale, thickness= thickness, text= None)#str(int(score[j])))

#================================================================
# Main starts here
#================================================================
parser = argparse.ArgumentParser(description='implementation of GUPNet')
parser.add_argument('--dataset', type=str, default = "kitti", help='one of kitti,nusc_kitti,nuscenes,waymo')
parser.add_argument('--folder' , type=str, default = "output/gup_nuscenes/results_test/data", help='evaluate model on validation set')
parser.add_argument('--folder2', type=str, default=  "output/prevolution_kitti_full/results_nusc_kitti_val/data")
parser.add_argument('--gt_folder',type=str, default = "", help='gt_folder')
parser.add_argument('--compression', type=int, default= 100)
parser.add_argument('--show_gt_in_image', action= 'store_true', default= False, help='show ground truth 3D boxes in image for sanity check')
args = parser.parse_args()


dataset  = args.dataset
folder   = args.folder
folder2  = args.folder2
compression_ratio = args.compression

show_ground_truth = True
show_baseline     = True
show_gt_in_image  = args.show_gt_in_image
video_demo_flag   = "video_demo" in args.folder
num_files_to_plot = 200

seed = 0
np.random.seed(seed)
random.seed(seed)
bev_scale    = 20   # Pixels per meter
bev_max_w    = 20   # Max along positive X direction. # This corresponds to the camera-view of (-max, max)
bev_w        = 2 * bev_max_w * bev_scale

# ==================================================================================================
# Dataset Specific Settings
# ==================================================================================================
if dataset == "kitti":
    box_class_list= ["car", "cyclist", "pedestrian"]
    lidar_points_in_gt = False

    if args.gt_folder != "":
        gt_folder = args.gt_folder
    elif video_demo_flag:
        gt_folder = "data/KITTI/video_demo"
    else:
        gt_folder = "data/KITTI/training"
    img_folder   = os.path.join(gt_folder, "image_2")
    cal_folder   = os.path.join(gt_folder, "calib")
    lab_folder   = os.path.join(gt_folder, "label_2")

    zfill_number = 6
    if video_demo_flag:
        print("Running with video_demo settings...")
        zfill_number = 10

    bev_max_z= 50
    ticks = [50, 40, 30, 20, 10, 0]

elif dataset == "nusc_kitti":
    box_class_list= ["car", "bicycle", "pedestrian", "barrier", "bus", "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck"]
    lidar_points_in_gt = True

    gt_folder = "data/nusc_kitti/validation"
    img_folder   = os.path.join(gt_folder, "image")
    cal_folder   = os.path.join(gt_folder, "calib")
    lab_folder   = os.path.join(gt_folder, "label")

    bev_max_z= 80
    ticks = [80, 70, 60, 50, 40, 30, 20, 10, 0]

elif dataset == "waymo":
    box_class_list= ["car", "cyclist", "pedestrian"]
    lidar_points_in_gt = True

    gt_folder = "data/waymo/validation"
    img_folder   = os.path.join(gt_folder, "image")
    cal_folder   = os.path.join(gt_folder, "calib")
    lab_folder   = os.path.join(gt_folder, "label")

    bev_max_z= 80
    ticks = [80, 70, 60, 50, 40, 30, 20, 10, 0]

elif dataset == "nuscenes":
    cls2id = {'barrier': 0, 'bicycle': 1, 'bus': 2, 'car': 3, 'construction_vehicle': 4, 'motorcycle': 5, 'pedestrian': 6, 'traffic_cone': 7, 'trailer': 8, 'truck': 9}
    id2cls = {v: k for k, v in cls2id.items()}
    gt_folder = None
    # load_pkl
    data_cache_path = os.path.join(os.getcwd(), 'nuscenes_val.pkl')
    data_cache      = pickle_read(file_path= data_cache_path)

    bev_w = int(1.6* bev_w)
    bev_max_z= 80
    ticks = [80, 70, 60, 50, 40, 30, 20, 10, 0]

print(box_class_list)

# ==================================================================================================
# Output
# ==================================================================================================
img_save_folder   = "images/qualitative/" + dataset #"waymo_with_gt_segment-14739149465358076158_4740_000_4760_000_with_camera_labels_new"
if video_demo_flag:
    img_save_folder += "/" + gt_folder.split("/")[-1]
os.makedirs(img_save_folder,exist_ok= True)

# plotting colors and other stuff
bev_c1       = (51, 51, 51)#(0, 250, 250)
bev_c2       = (255, 255, 255)#(0, 175, 250)
c_gts        = (10, 175, 10)
c            = (255,51,153)#(255,48,51)#(255,0,0)#(114,211,254) #(252,221,152 # RGB

color_gt     = (153,255,51)#(0, 255 , 0)
color_pred_2 = (59, 221, 255)#(51,153,255)#(94,45,255)#(255, 128, 0)
use_classwise_color = True

files_list   = sorted(glob.glob(folder + "/*.txt"))
num_files    = len(files_list)

print("Choosing {} files out of {} files...".format(num_files_to_plot, num_files))

if video_demo_flag:
    file_index = np.arange(num_files)
else:
    file_index = np.sort(np.random.choice(range(num_files), num_files_to_plot, replace=False))


# if dataset == "kitti":
#     file_index = [669, 983, 1022, 1198, 1232, 1355]
# elif dataset == "nusc_kitti":
#     file_index = [2268, 828, 451, 5247, 5129, 4399, 3930]
# elif dataset == "waymo":
#     file_index = [689, 1074, 4061, 8694, 11314, 12874]
# file_index = np.array(file_index)

for i in range(file_index.shape[0]):
    curr_index = file_index[i]
    pred_file  = files_list[curr_index]
    basename   = os.path.basename(pred_file).split(".")[0]

    # basename   = str(file_index[i]).zfill(6)
    # pred_file  = os.path.join(args.folder, basename + ".txt")

    if dataset == "nuscenes":
        #match_by_file_name
        matched_index = [t for t in range(len(data_cache)) if (os.path.basename(data_cache[t]['file_name']).split(".")[0] ==  basename)]
        matched_index = matched_index[0]
        img_file   = data_cache[matched_index]['file_name']

        p2_temp    = np.array(data_cache[matched_index]['intrinsics']).reshape(3,3)
        p2         = np.eye(4)
        p2[:3, :3] = p2_temp

        annotations     = data_cache[matched_index]['annotations']
        num_annotations = len(annotations)
        lines = []
        for i, anno_curr in enumerate(annotations):
            category_name = id2cls[anno_curr['category_id']]
            cx, cy, cz    = anno_curr['box3d']['center']
            w, l, h       = anno_curr['box3d']['wlh']
            ry3d          = anno_curr['box3d']['yaw']
            alpha         = convertRot2Alpha(ry3d= ry3d, z3d= cz, x3d= cx)
            x1, y1, x2, y2= anno_curr['box2d']
            # Label format as KITTI
            #   0   1    2     3   4   5  6    7    8    9    10   11   12         13    14   15
            # (cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d + h/2, z3d, ry3d, score,
            line = [category_name, 0, 0, alpha, x1, y1, x2, y2, h, w, l, cx, cy + h/2.0, cz, ry3d]
            lines.append(line)
        gt_img = pd.DataFrame(lines).values

    else:
        # if dataset == "kitti":
            # if not video_demo_flag and int(basename) not in   [35, 5086, 4485, 3207, 1868, 1101, 3135]:
            #     continue

        cal_file   = os.path.join(cal_folder, basename + ".txt")
        img_file   = os.path.join(img_folder, basename + ".png")
        label_file = os.path.join(lab_folder, basename + ".txt")
        p2         = get_calib_from_file(cal_file)['P2']
        gt_img     = read_csv(label_file, ignore_warnings= True, use_pandas= True)

        if gt_img is not None:
            gt_other      = gt_img[:, 1:].astype(float)
            #       0  1   2      3   4   5   6    7    8    9   10   11   12    13     14
            # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, lidar
            if lidar_points_in_gt:
                gt_bad_index  = np.logical_or(np.logical_or(np.logical_or(gt_other[:, 14] == 0, gt_other[:, 3]-gt_other[:, 5] >=0), gt_other[:, 4]-gt_other[:, 6] >=0), gt_other[:, 12] <=0 )
            else:
                gt_bad_index  = np.logical_or(np.logical_or(gt_other[:, 3]-gt_other[:, 5] >=0, gt_other[:, 4]-gt_other[:, 6] >=0), gt_other[:, 12] <=0 )

            gt_good_index = np.logical_not(gt_bad_index)
            gt_img        = gt_img[gt_good_index]

    img        = imread(img_file)

    canvas_bev = create_colorbar(bev_max_z * bev_scale, bev_w, color_lo=bev_c1, color_hi=bev_c2)

    if show_gt_in_image:
        plot_boxes_on_image_and_in_bev(gt_img, plot_color= color_gt, box_class_list= box_class_list, use_classwise_color= use_classwise_color, show_3d= True)
    else:
        if show_ground_truth:
            plot_boxes_on_image_and_in_bev(gt_img, plot_color= color_gt, box_class_list= box_class_list, show_3d= False)

    if show_baseline:
        predictions_file_2     = os.path.join(folder2, basename + ".txt")
        predictions_img_2      = read_csv(predictions_file_2, ignore_warnings= True, use_pandas= True)
        plot_boxes_on_image_and_in_bev(predictions_img_2, plot_color= color_pred_2, box_class_list= box_class_list, show_3d= False, thickness= 8)

    if not show_gt_in_image:
        predictions_img  = read_csv(pred_file, ignore_warnings= True, use_pandas= True)
        if predictions_img is not None and gt_img is not None:
            plot_boxes_on_image_and_in_bev(predictions_img, plot_color = c, box_class_list= box_class_list, use_classwise_color= use_classwise_color, thickness= 6)
            print("Predictions = {} GT= {}".format(predictions_img.shape[0], gt_img.shape[0]))

    canvas_bev = cv2.flip(canvas_bev, 0)
    # draw tick marks
    draw_tick_marks(canvas_bev, ticks)

    im_concat = imhstack(img, canvas_bev)
    save_path = os.path.join(img_save_folder, basename + ".png")
    print("Saving to {} with compression ratio {}".format(save_path, compression_ratio))
    imwrite(im_concat, save_path)

    img_concat_small = imutils.resize(im_concat, height=96)
    img_concat_small = im_concat
    # plt.imshow(img_concat_small[:,:,::-1])
    # plt.title(basename)
    # plt.show()

    if compression_ratio != 100:
        # Save smaller versions of image
        command = "convert -resize " + str(compression_ratio) + "% " + save_path + " " + save_path
        os.system(command)

    pass

# Video conversion
# r= framerate
# ffmpeg -safe 0 -r 10 -f concat -i file -q:v 1 -r 10 -pix_fmt yuv420p images/out.mp4
# /snap/bin/ffmpeg -safe 0 -r 25 -f concat -i file -q:v 1 -r 25 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx265 images/equivariance_error_demo.mp4
