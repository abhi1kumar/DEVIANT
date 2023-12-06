"""
    Sample Run:
    python plot/plot_equivariance_error_for_raw_sequences.py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn

np.set_printoptions   (precision= 4, suppress= True)
torch.set_printoptions(precision= 4, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

from lib.helpers.file_io import *
from lib.helpers.util import get_downsampled_image, get_bounds, mkdir_if_missing
from numpy.linalg import norm

def get_img_features(model, scale_name_list, level):
    output = []
    for i, scale_name in enumerate(scale_name_list):
        scale_npy_path   = os.path.join("output/" + model, rel_path + "/val_scale/level_" + str(level) + ".npy")
        data             = read_numpy(scale_npy_path).astype(np.float32)
        output.append(data)

    return output


def get_error(ref_image, prev_image, masking= False):
    input_base_cnn        = ref_image[np.newaxis, : , :]
    scaled_input_base_cnn = get_downsampled_image(img= input_base_cnn, curr_scale= curr_scale, height_first_flag= False)[0]
    input2_cnn            = prev_image
    if masking:
        left_x, right_x, top_y, bottom_y = get_bounds(img_shape_x= input_base_cnn.shape[2], img_shape_y= input_base_cnn.shape[1], curr_scale= curr_scale)
        scaled_input_base_cnn = scaled_input_base_cnn[top_y:bottom_y, left_x:right_x]
        input2_cnn            = input2_cnn[top_y:bottom_y, left_x:right_x]

    error                     = norm(scaled_input_base_cnn - input2_cnn)/(norm(scaled_input_base_cnn) + 0.01)
    error_full_img            = np.abs(scaled_input_base_cnn - input2_cnn)/(np.abs(scaled_input_base_cnn) + 0.01)

    return error, error_full_img

model_cnn     = "config_run_201_a100_v0_1"
model_deviant = "run_221"

level         = 2
channel_ind   = 19
key           = ["video_demo_3"]
frame_diff    = 3
dpi           = 150
curr_scale    = 1 + (1.7 * frame_diff / 10.0)
vmin          = 0
vmax          = 6
alpha         = 1.0
fs            = 30
eps           = 0.001
cmap          = diverge_map(params.color_set1_pink/255.0, None)

rel_path      = "result_" + key[0]
img_folder    = "data/KITTI/" + key[0] + "/image_2"
output_folder = "images/equivariance_error/" + key[0]
mkdir_if_missing(output_folder, delete_if_exist= True)

img_path_list = sorted(glob.glob(os.path.join(img_folder, "*.png")))
print(img_path_list[:5])

output_cnn     =  get_img_features(model= model_cnn    , scale_name_list= key, level= level)
output_deviant =  get_img_features(model= model_deviant, scale_name_list= key, level= level)

num_frames = output_cnn[0].shape[0]
print("Shapes of feature maps")
print(output_cnn[0].shape)
print(output_deviant[0].shape)

for i in range(frame_diff, num_frames):
    ref_image_cnn      = output_cnn[0][i, channel_ind]
    prev_image_cnn     = output_cnn[0][i, channel_ind]
    _, error_full_cnn  = get_error(ref_image_cnn, prev_image_cnn)

    ref_image_deviant     = output_deviant[0][i, channel_ind]
    prev_image_deviant    = output_deviant[0][i, channel_ind]
    _, error_full_deviant = get_error(ref_image_deviant, prev_image_deviant)

    error_full_cnn        = np.log(error_full_cnn  + eps)
    error_full_deviant    = np.log(error_full_deviant + eps)

    fig = plt.figure(figsize= (18,9), dpi= dpi)
    matplotlib.rcParams.update({'font.size': fs})
    plt.subplot(311)
    img = imread(img_path_list[i])
    plt.imshow(img[:,:,::-1])
    plt.axis('off')
    plt.text(-220, 220, 'Video', fontsize= fs)

    plt.subplot(312)
    plt.imshow(error_full_cnn, vmin= vmin, vmax= vmax, cmap= cmap, alpha= alpha)
    plt.xticks([])
    plt.yticks([])
    plt.text(-70, 55, 'Vanilla', fontsize= fs)

    plt.subplot(313)
    plt.imshow(error_full_deviant, vmin= vmin, vmax= vmax, cmap= cmap, alpha= alpha)
    plt.xticks([])
    plt.yticks([])
    plt.text(-80, 50, 'Proposed', fontsize= fs)

    cax = plt.axes([0.76, 0.25, 0.02, 0.48])
    cbar = plt.colorbar(cax=cax)
    cbar_ticks = np.around(np.array([0, vmax/4,vmax/2, 3*vmax/4, vmax-0.1]), decimals= 1)
    cbar.set_ticks(cbar_ticks)
    fig.tight_layout(h_pad= 0.2, w_pad= 0.2)

    savefig(plt, output_folder + "/" + os.path.basename(img_path_list[i]))

    # plt.show()
    plt.close()