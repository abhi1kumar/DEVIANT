"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import cv2

np.set_printoptions   (precision= 4, suppress= True)
torch.set_printoptions(precision= 4, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from skimage.measure import compare_ssim as struct_sim

from lib.projective.log_polar_conv import convert_HWC_to_CHW, convert_CHW_to_HWC, \
    cartesian_to_log_polar_feature_map, log_polar_to_cartesian_feature_map, our_interpolate
from lib.helpers.file_io import *
from lib.datasets.kitti_utils import Calibration

def get_error_stats(input1_hwc, input2_hwc, prefix="", ignore_black_area= False):
    if type(input1_hwc) == np.ndarray:
        if ignore_black_area:
            h, w, c = input1_hwc.shape
            # Ignore the black areas since
            final_w = int(0.9 * w)
            t1 = input1_hwc[:, :final_w]
            t2 = input2_hwc[:, :final_w]
        else:
            t1 = input1_hwc
            t2 = input2_hwc

        l2   = np.linalg.norm(t1 - t2)/ np.prod(t1.shape)
        linf = np.max(np.abs(t1 - t2))
        ssim = struct_sim(t1, t2, multichannel= True)

    return ssim, l2, linf

def numpy_log_polar(input_np_height_curr, center_orig, h_orig, upsample= 1.0):
    """
    Image can be in arbitray resolution
    orig = original resolution
    """
    h_curr, w_curr, c_curr = input_np_height_curr.shape
    center_curr            = center_orig * float(h_curr)/h_orig

    if upsample != 1.0:
        # Upsample image and center
        input_np_height_curr  = cv2.resize(input_np_height_curr, (int(w_curr * upsample), int(h_curr * upsample)))
        center_curr        = center_curr * upsample

    input_np_channel_curr     = convert_HWC_to_CHW(input_np_height_curr)
    log_polar_np_channel      = cartesian_to_log_polar_feature_map(input_np_channel_curr, center_cartesian_feature_map= center_curr)
    input_back_np_channel     = log_polar_to_cartesian_feature_map(log_polar_np_channel, center_cartesian_feature_map= center_curr)
    input_back_np_height      = convert_CHW_to_HWC(input_back_np_channel)

    if upsample != 1.0:
        # Downsample, cv2.resize operates in Height space
        input_back_np_height = cv2.resize(input_back_np_height, (w_curr, h_curr))

    return log_polar_np_channel, input_back_np_height

def pytorch_log_polar(input_ten_channel_curr, center_ten_orig, h_orig, upsample_factor_for_interpolate= 1.0, mode="bilinear", upsample_factor_for_log_polar_output= 1.0, interpolation_method="deterministic", debug= False):

    _, h_curr, w_curr, c_curr = input_ten_channel_curr.shape
    center_ten_curr        = center_ten_orig * float(h_curr) / h_orig

    if upsample_factor_for_interpolate != 1.0:
        input_ten_channel_curr  = our_interpolate(input_ten_channel_curr, scale_factor= upsample_factor_for_interpolate, mode= mode)
        center_ten_curr         = center_ten_curr * upsample_factor_for_interpolate

    log_polar_ten_channel  = cartesian_to_log_polar_feature_map(input_ten_channel_curr, center_cartesian_feature_map= center_ten_curr, interpolation_method= interpolation_method, upsample_factor_for_log_polar_output= upsample_factor_for_log_polar_output)
    input_back_ten_channel = log_polar_to_cartesian_feature_map(log_polar_ten_channel, center_cartesian_feature_map= center_ten_curr, interpolation_method= interpolation_method, upsample_factor_for_log_polar_output= upsample_factor_for_log_polar_output)

    if upsample_factor_for_interpolate != 1.0:
        input_back_ten_channel = our_interpolate(input_back_ten_channel, scale_factor= 1.0/upsample_factor_for_interpolate, mode= mode)

    # # Take the first image for display
    # log_polar_map_ten_disp = convert_CHW_to_HWC(log_polar_ten_hwc[0].squeeze(0).numpy()).astype(np.uint8)
    # input_map_back_ten_disp= convert_CHW_to_HWC(input_back_ten_height[0].squeeze(0).numpy()).astype(np.uint8)

    input_back_ten_height = convert_CHW_to_HWC(input_back_ten_channel)

    if debug:
        # print("---------------------------------------")
        print("Tensor I/P O/P shape interpol= {}".format(interpolation_method))
        print(input_ten_channel_curr.shape)
        # print(log_polar_map_ten_disp.shape)
        # print(input_map_back_ten_disp.shape)

    # return log_polar_ten_hwc, log_polar_map_ten_disp, input_back_ten_height, input_map_back_ten_disp

    return log_polar_ten_channel, input_back_ten_height


#===================================================================================================
# Main starts here
#===================================================================================================
root_dir      = "data/KITTI"
split         = "val"

modes_list    = ["numpy", "bilinear", "nearest"]
fixed_res     = (384, 1280)
input_res_list= [12, 24, 48, 96, 192, 384]
upsample_list = [1.0, 2.0, 4.0]
fs            = 24
matplotlib.rcParams.update({'font.size': fs})
batch         = 1
fig_lw        = params.lw - 2
legend_fs     = params.legend_fs
alpha         = 0.15
legend_border_axes_plot_border_pad = params.legend_border_axes_plot_border_pad
legend_border_pad                  = params.legend_border_pad
legend_vertical_label_spacing      = params.legend_vertical_label_spacing
legend_marker_text_spacing         = params.legend_marker_text_spacing


num_modes     = len(modes_list)
num_input_res = len(input_res_list)
num_upsample  = len(upsample_list)


split_file = os.path.join(root_dir, 'ImageSets', split + '.txt')
idx_list   = [x.strip() for x in open(split_file).readlines()]
num_images    = len(idx_list)

ssim_stats = np.zeros((num_modes, num_upsample, num_input_res, 3))

"""
for i, mode in enumerate(modes_list):
    for j, upsample in enumerate(upsample_list):
        for k, input_res in enumerate(input_res_list):

            ssim_val = []

            for t, idx in enumerate(idx_list):
                img_file_path   = os.path.join(root_dir, "training/image_2", idx + ".png")
                calib_file_path = os.path.join(root_dir, "training/calib"  , idx + ".txt")

                img_height    = imread(img_file_path)
                calib  = Calibration(calib_file_path)
                center = np.array([calib.cu, calib.cv])

                h_orig, w_orig, c_orig = img_height.shape
                scale_curr = input_res/float(h_orig)

                img_height_curr = cv2.resize(img_height, (int(w_orig * scale_curr), int(h_orig * scale_curr)))

                if mode == "numpy":
                    _, img_back_np_height  = numpy_log_polar(img_height_curr, center_orig = center, h_orig= h_orig, upsample= upsample)
                else:
                    img_ten_channel_curr = convert_HWC_to_CHW(torch.from_numpy(img_height_curr).float().unsqueeze(0).repeat(batch, 1, 1, 1))
                    center_ten_orig  = torch.from_numpy(center).float().unsqueeze(0).repeat(batch, 1)
                    _, img_back_ten_height = pytorch_log_polar(
                        input_ten_channel_curr= img_ten_channel_curr, center_ten_orig= center_ten_orig, h_orig= h_orig, upsample_factor_for_interpolate= upsample, mode= mode)

                    img_back_np_height  = img_back_ten_height.cpu().numpy()[0]

                ssim, _, _ =  get_error_stats(input1_hwc= img_height_curr.astype(np.float32), input2_hwc= img_back_np_height.astype(np.float32))
                ssim_val.append(ssim)

                if t >= 2:
                    break

            # Get the stats
            ssim_val = np.array(ssim_val)
            # print(ssim_val.shape)
            ssim_stats[i, j, k, 0] = np.mean(ssim_val)
            ssim_stats[i, j, k, 1] = np.std(ssim_val)
            ssim_stats[i, j, k, 2] = ssim_val.shape[0]

            print(mode, upsample, input_res, ssim_stats[i, j, k])

save_path = "images/ssim_stats_for_log_polar.npy"
print("Saving to {}".format(save_path))
save_numpy(save_path, ssim_stats)
"""
ssim_stats = read_numpy(path= "images/ssim_stats_for_upsample_log_polar_bkp.npy")

# Now plot different values
input_res_np = np.array(input_res_list)
style_list = ['dotted', 'dashed', 'solid']
color_list = [params.color_set1_cyan/255.0, params.color_set1_pink/255.0, params.color_set1_yellow/255.0]
# color_list = [params.color_red/255.0, params.color_blue/255.0, params.color_set1_pink/255.0]
# color_list  = ["red", "orange", "purple"]
plt.figure(dpi= params.DPI)
for i in [2, 1]:
    for j in range(num_upsample):
        mean_plot   = ssim_stats[i, j, :, 0]
        std_dev_plot= ssim_stats[i, j, :, 1]
        if j == 2:
            label_text = modes_list[i].title() #+ " x" + str(int(upsample_list[j]))
            plt.plot(input_res_np, mean_plot, linestyle= style_list[j], color= color_list[i], lw= fig_lw, label= label_text)
        else:
            plt.plot(input_res_np, ssim_stats[i, j, :, 0], linestyle= style_list[j], color= color_list[i], lw= fig_lw)
        plt.fill_between(input_res_np, mean_plot - std_dev_plot, mean_plot + std_dev_plot, color= color_list[i], alpha= alpha)

# Make a custom legend
# patch1 = mpatches.Patch(color= color_list[0], label= modes_list[0].title())
patch_dummy = Line2D([0], [0], linestyle= 'none')
patch2 = mpatches.Patch(color= color_list[1], label= "PyTorch " + modes_list[1].title())
patch3 = mpatches.Patch(color= color_list[2], label= "PyTorch " + modes_list[2].title())
patch4 = Line2D([0], [0], color= "black", lw= fig_lw, label= "Up x" +  str(int(upsample_list[0])), linestyle= style_list[0])
patch5 = Line2D([0], [0], color= "black", lw= fig_lw, label= "Up x" +  str(int(upsample_list[1])), linestyle= style_list[1])
patch6 = Line2D([0], [0], color= "black", lw= fig_lw, label= "Up x" +  str(int(upsample_list[2])), linestyle= style_list[2])

# legend_handles = [patch1]
legend_handles = []
legend_handles.append(patch2)
legend_handles.append(patch3)
legend_handles.append(patch_dummy)
legend_handles.append(patch4)
legend_handles.append(patch5)
legend_handles.append(patch6)


plt.ylim(0, 1)
plt.grid(True)
plt.xlabel('Resolution')
plt.ylabel('SSIM')
plt.legend(handles= legend_handles, ncol=2, loc='lower right', fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
ax = plt.gca()
ax.set_xscale('log')
ax.set_xticks(input_res_list)
# ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax.tick_params(axis=u'both', which=u'both',length=0)
ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticklabels(input_res_list)

# locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
# ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

path = "images/ssim_for_upsample_log_polar.png"
savefig(plt, path)
plt.show()
plt.close()