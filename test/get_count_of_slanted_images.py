"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn

np.set_printoptions   (precision= 4, suppress= True)
torch.set_printoptions(precision= 4, sci_mode= False)

import glob
from lib.helpers.util import is_slanted

def get_slanted_images_cnt(label_folder, ego_height= 1.65):
    print("\n{}".format(label_folder))
    files_list   = sorted(glob.glob(label_folder + "/*.txt"))
    slant_cnt = 0

    for i, label_file in enumerate(files_list):

        if is_slanted(label_file, ego_height):
            # print(label_file)
            slant_cnt += 1

        if (i + 1) % 1000 == 0 or i == len(files_list)-1:
            print("{} in {} images done".format(slant_cnt, i+1))

    return slant_cnt, len(files_list)

# KITTI train + validation
label_folder = "data/KITTI/training/label_2/"
ego_height= 1.65
slant_cnt, tot_cnt = get_slanted_images_cnt(label_folder, ego_height= ego_height)
print("KITTI Train+Val Slant= {} / Total = {}".format(slant_cnt, tot_cnt))


# nusc_kitti validation
# They use Renault Zoe supermini electric cars in CVPR paper
# https://en.wikipedia.org/wiki/Renault_Zoe
label_folder = "/media/abhinav/baap/abhinav/datasets/nusc_kitti_front/val/label_2"
ego_height   = 1.56
slant_cnt, tot_cnt = get_slanted_images_cnt(label_folder, ego_height= ego_height)
print("Nusc_kitti Val Slant= {} / Total = {}".format(slant_cnt, tot_cnt))


# Waymo validation folder
# https://github.com/waymo-research/waymo-open-dataset/issues/92#issuecomment-573355327
# 70 inches = 1.78 m
label_folder = "data/waymo/validation/label/"
ego_height= 1.78
# slant_cnt, tot_cnt = get_slanted_images_cnt(label_folder, ego_height= ego_height)
print("Waymo Open Val Slant= {} / Total = {}".format(slant_cnt, tot_cnt))

#KITTI Train+Val Slant= 392 / Total = 7481
#Nusc_kitti Val Slant= 1252 / Total = 6019
#Waymo Open Val Slant= 12111 / Total = 39848