"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn

from numpy.linalg import norm

np.set_printoptions   (precision= 4, suppress= True)
torch.set_printoptions(precision= 4, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib
from lib.helpers.file_io import read_numpy
from lib.helpers.util import get_downsampled_image

def get_equivariance_error(input, input2, scale= 1):
    if input.dtype == 'float16':
        input  = input.astype(np.float)
    if input2.dtype == 'float16':
        input2 = input2.astype(np.float)

    # Downsample the input first by the scale
    scaled_input = get_downsampled_image(img= input, curr_scale= scale, height_first_flag= False)
    error        = norm(input2 - scaled_input)/(norm(scaled_input) + 0.01)

    return error

def get_stats_for_model(model, scale_1  = "scale_1", scale_ran= "random_scaling", level= 5):
    rel_path         = "results_kitti_val"
    scale_1_path     = os.path.join("output/" + model, rel_path + "/" + scale_1   + "/level_" + str(level) + ".npy")
    scale_ran_path   = os.path.join("output/" + model, rel_path + "/" + scale_ran + "/level_" + str(level) + ".npy")

    # print("")
    # print(scale_1_path)

    mapping_pkl_path = 'kitti_' + scale_ran + '.pkl'
    mapping          = pickle_read(mapping_pkl_path, show_message= False)

    scale_1_data     = read_numpy(scale_1_path)
    scale_ran_data   = read_numpy(scale_ran_path)

    num_images       = scale_1_data.shape[0]
    equiv_error_img  = []

    for i in range(num_images):
        curr_scale = mapping[i][2]
        curr_error = get_equivariance_error(scale_1_data[i], scale_ran_data[i], scale= curr_scale)
        if curr_error > 100:
            continue
        equiv_error_img.append(curr_error)

    equiv_error_img   = np.array(equiv_error_img)
    print("Model= {:20s} Level= {} N= {} Mean= {:.2f} Std= {:.2f}".\
          format(model, level, equiv_error_img.shape[0], np.mean(equiv_error_img), np.std(equiv_error_img)))

    return  np.array([equiv_error_img.shape[0], np.mean(equiv_error_img), np.std(equiv_error_img)])

# =====================================================================
# Main starts here
# =====================================================================
model2   = "config_retrain_v100"
model1   = "run_11"


color_list = [params.color_set1_pink/255.0, params.color_set1_cyan/255.0]
label_list = ['DEVIANT', 'GUP Net']
ylabel_txt = r'Log Eqv Error $(\ln \Delta)$'
ylim_arr   = (0, 11)
alpha      = 0.5
fig_lw     = params.lw
legend_border_axes_plot_border_pad = params.legend_border_axes_plot_border_pad
legend_border_pad                  = params.legend_border_pad
legend_vertical_label_spacing      = params.legend_vertical_label_spacing
legend_marker_text_spacing         = params.legend_marker_text_spacing


# get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "random_scaling", level= 5)
print("=============================================")
# get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "random_scaling", level= 0)
# get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "random_scaling", level= 1)
# get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "random_scaling", level= 2)
# get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "random_scaling", level= 3)
# get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "random_scaling", level= 4)
# get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "random_scaling", level= 5)

print("=============================================")
# get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "random_scaling", level= 0)
# get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "random_scaling", level= 1)
# get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "random_scaling", level= 2)
# get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "random_scaling", level= 3)
# get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "random_scaling", level= 4)
# get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "random_scaling", level= 5)


model1_mean = np.array([2.17, 2.11, 2.00, 2.27, 2.07])
model1_std  = np.array([0.59, 0.49, 0.44, 0.55, 0.60])
model2_mean = np.array([2.54, 4.45, 3.78, 4.66, 8.22])
model2_std  = np.array([0.66, 1.22, 0.81, 1.21, 3.45])

x_arr       = np.arange(model1_mean.shape[0]) + 1


plt.figure(figsize= params.size, dpi= params.DPI)
plt.plot(x_arr, model1_mean, color= color_list[0], lw= fig_lw, label= label_list[0])
plt.plot(x_arr, model2_mean, color= color_list[1], lw= fig_lw, label= label_list[1])
plt.fill_between(x_arr, model1_mean - model1_std, model1_mean + model1_std, color= color_list[0], alpha= alpha)
plt.fill_between(x_arr, model2_mean - model2_std, model2_mean + model2_std, color= color_list[1], alpha= alpha)
plt.ylabel(ylabel_txt)
plt.xlabel('Block of DLA-34')
plt.xlim((1.0, 5.0))
plt.ylim(ylim_arr)
plt.grid()
plt.legend(loc= 'upper left', borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
savefig(plt, "images/equivariance_error_at_blocks.png")
# plt.show()


# =================================================================================================
# With downsampling factor
# =================================================================================================
level_to_plot  = 2
num_models     = 2
scales_list    = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
all_stats      = np.zeros((num_models, scales_list.shape[0], 3)) # N, mean, stats

# all_stats[0,0] = get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "down_1_2", level= level_to_plot)
# all_stats[0,1] = get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "down_1_4", level= level_to_plot)
# all_stats[0,2] = get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "down_1_6", level= level_to_plot)
# all_stats[0,3] = get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "down_1_8", level= level_to_plot)
# all_stats[0,4] = get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "down_2", level= level_to_plot)
#
# all_stats[1,0] = get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "down_1_2", level= level_to_plot)
# all_stats[1,1] = get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "down_1_4", level= level_to_plot)
# all_stats[1,2] = get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "down_1_6", level= level_to_plot)
# all_stats[1,3] = get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "down_1_8", level= level_to_plot)
# all_stats[1,4] = get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "down_2", level= level_to_plot)

# Model= run_11               Level= 2 N= 312 Mean= 1.51 Std= 0.08
# Model= run_11               Level= 2 N= 312 Mean= 1.77 Std= 0.09
# Model= run_11               Level= 2 N= 312 Mean= 2.00 Std= 0.10
# Model= run_11               Level= 2 N= 312 Mean= 2.44 Std= 0.11
# Model= run_11               Level= 2 N= 312 Mean= 2.91 Std= 0.14
# Model= config_retrain_v100  Level= 2 N= 312 Mean= 3.04 Std= 0.22
# Model= config_retrain_v100  Level= 2 N= 312 Mean= 3.35 Std= 0.24
# Model= config_retrain_v100  Level= 2 N= 312 Mean= 3.69 Std= 0.28
# Model= config_retrain_v100  Level= 2 N= 312 Mean= 4.44 Std= 0.35
# Model= config_retrain_v100  Level= 2 N= 312 Mean= 5.99 Std= 0.56

all_stats[:,:,0] = 312
all_stats[0,0,1] = 1.51
all_stats[0,0,2] = 0.08
all_stats[0,1,1] = 1.77
all_stats[0,1,2] = 0.09
all_stats[0,2,1] = 2.00
all_stats[0,2,2] = 0.10
all_stats[0,3,1] = 2.44
all_stats[0,3,2] = 0.11
all_stats[0,4,1] = 2.91
all_stats[0,4,2] = 0.14


all_stats[1,0,1] = 3.04
all_stats[1,0,2] = 0.22
all_stats[1,1,1] = 3.35
all_stats[1,1,2] = 0.24
all_stats[1,2,1] = 3.69
all_stats[1,2,2] = 0.28
all_stats[1,3,1] = 4.44
all_stats[1,3,2] = 0.35
all_stats[1,4,1] = 5.99
all_stats[1,4,2] = 0.56

x_arr       = scales_list

plt.figure(figsize= params.size, dpi= params.DPI)
plt.plot(x_arr, all_stats[0, :, 1], color= color_list[0], lw= fig_lw, label= label_list[0])
plt.plot(x_arr, all_stats[1, :, 1], color= color_list[1], lw= fig_lw, label= label_list[1])
plt.fill_between(x_arr, all_stats[0, :, 1] - all_stats[0, :, 2], all_stats[0, :, 1] + all_stats[0, :, 2], color= color_list[0], alpha= alpha)
plt.fill_between(x_arr, all_stats[1, :, 1] - all_stats[1, :, 2], all_stats[1, :, 1] + all_stats[1, :, 2], color= color_list[1], alpha= alpha)
plt.ylabel(ylabel_txt)
plt.xlabel(r'Image DownScaling Factor $(1/s)$')
plt.xlim((1.2, 2.0))
plt.ylim(ylim_arr)
plt.grid()
plt.legend(loc= 'upper left', borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
savefig(plt, "images/equivariance_error_with_scaling.png")

# =================================================================================================
# On videos
# =================================================================================================
level_to_plot  = 2
num_models     = 2
scales_list    = np.array([1, 2, 3])
all_stats      = np.zeros((num_models, scales_list.shape[0], 3)) # N, mean, stats

# all_stats[0,0] = get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "past_1", level= level_to_plot)
# all_stats[0,1] = get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "past_2", level= level_to_plot)
# all_stats[0,2] = get_stats_for_model(model1, scale_1  = "scale_1", scale_ran= "past_3", level= level_to_plot)
#
# all_stats[1,0] = get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "past_1", level= level_to_plot)
# all_stats[1,1] = get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "past_2", level= level_to_plot)
# all_stats[1,2] = get_stats_for_model(model2, scale_1  = "scale_1", scale_ran= "past_3", level= level_to_plot)

# Model= run_11               Level= 2 N= 312 Mean= 1.77 Std= 0.11
# Model= run_11               Level= 2 N= 312 Mean= 1.96 Std= 0.11
# Model= run_11               Level= 2 N= 312 Mean= 2.13 Std= 0.11
# Model= config_retrain_v100  Level= 2 N= 312 Mean= 3.13 Std= 0.23
# Model= config_retrain_v100  Level= 2 N= 312 Mean= 3.42 Std= 0.26
# Model= config_retrain_v100  Level= 2 N= 312 Mean= 3.69 Std= 0.28

all_stats[:,:,0] = 312
all_stats[0,0,1] = 1.77
all_stats[0,0,2] = 0.11
all_stats[0,1,1] = 1.96
all_stats[0,1,2] = 0.11
all_stats[0,2,1] = 2.13
all_stats[0,2,2] = 0.11

all_stats[1,0,1] = 3.13
all_stats[1,0,2] = 0.23
all_stats[1,1,1] = 3.42
all_stats[1,1,2] = 0.26
all_stats[1,2,1] = 3.69
all_stats[1,2,2] = 0.28

x_arr       = scales_list

plt.figure(figsize= params.size, dpi= params.DPI)
plt.plot(x_arr, all_stats[0, :, 1], color= color_list[0], lw= fig_lw, label= label_list[0])
plt.plot(x_arr, all_stats[1, :, 1], color= color_list[1], lw= fig_lw, label= label_list[1])
plt.fill_between(x_arr, all_stats[0, :, 1] - all_stats[0, :, 2], all_stats[0, :, 1] + all_stats[0, :, 2], color= color_list[0], alpha= alpha)
plt.fill_between(x_arr, all_stats[1, :, 1] - all_stats[1, :, 2], all_stats[1, :, 1] + all_stats[1, :, 2], color= color_list[1], alpha= alpha)
plt.ylabel(ylabel_txt)
plt.xlabel(r'Frame')
plt.gca().set_xticks(x_arr)
plt.gca().set_xticklabels(['t-1', 't-2', 't-3'])
plt.xlim((1, 3))
plt.ylim(ylim_arr)
plt.grid()
plt.legend(loc= 'upper left', borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
savefig(plt, "images/equivariance_error_with_video_frames.png")