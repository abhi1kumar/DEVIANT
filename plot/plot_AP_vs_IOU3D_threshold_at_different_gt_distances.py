

"""
    Sample Run:
    python plot/plot_AP_vs_IOU3D_threshold_at_different_gt_distances.py
    python plot/plot_AP_vs_IOU3D_threshold_at_different_gt_distances.py -i output/kitti_acceptance_prob_overlap_lr_3e-3_freeze_hard_4_finetune_6/results/results_test
    python plot/plot_AP_vs_IOU3D_threshold_at_different_gt_distances.py -i output/kitti_acceptance_prob_foregrounds_lr_4e-3_hard_4/results/results_test_pred_times_class/ output/kitti_acceptance_prob_kinematic_lr_4e-3/results/results_test output/kitti_acceptance_prob_overlap_lr_3e-3_freeze_hard_4_finetune_6/results/results_test_predicted/ -l Ours Self_Balancing Ours_old

    Plots AP3D with IOU3D threshold for boxes at different less than ground truth distances.
"""
# =======================================--
# python modules
# =======================================--
import os, sys
sys.path.append(os.getcwd())

import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib

# =======================================
# custom modules
# =======================================
from plot.common_operations import *

file_path_basename = "AP_vs_IOU3D_threshold_at_different_gt_distances.pkl"
legend_fs = params.legend_fs+4
fig_size  = (8,5)#params.size
fig_lw    = params.lw - 2
pad_inches= 0.04
legend_border_axes_plot_border_pad = params.legend_border_axes_plot_border_pad
legend_border_pad                  = params.legend_border_pad
legend_vertical_label_spacing      = params.legend_vertical_label_spacing
legend_marker_text_spacing         = params.legend_marker_text_spacing
#===============================================================================
# Argument Parsing
#===============================================================================
ap      = argparse.ArgumentParser()
ap.add_argument('-i', '--input_folder', nargs='+', default= ["output/run_221/result_kitti_val"], help= 'path of the input folder')
ap.add_argument('-l', '--labels'      , nargs='+', default= ["DEVIANT"], help= 'labels of the input folder')
args    = ap.parse_args()

#=======================================================================================
# load both results
#=======================================================================================
folder1 = "output/config_run_201_a100_v0_1/result_kitti_val"
label_folder1 = "GUP Net"
iou_keys    = ["0_3", "0_4"   , "0_5", "0_6", "0_7"]
iou_vals    = [0.3  , 0.4     , 0.5, 0.6, 0.7]
top_ylim  = 1
bottom_ylim= -0.02


# folder1 = "output/groumd_nms_dpp/results/results_dpp_some_500_alpha_1"
# label_folder1 = "DPP(500,a=1)"
# iou_keys    = ["0_5", "0_6", "0_7"]
# iou_vals    = [0.5, 0.6, 0.7]
# top_ylim  = 0.75
# bottom_ylim= 0.03

folder2 = args.input_folder
num_input_folders = len(folder2)
print("Number of models to plot = {}".format(num_input_folders))

m3d_res = pickle_read(os.path.join(folder1, file_path_basename))
our_res = []
for j in range(num_input_folders):
    our_res.append(pickle_read(os.path.join(folder2[j], file_path_basename)))

dis_keys    = ["15" , "30"    , "45", "60"]
colors_list = ["red", "orange", "purple", "blue"]

# For number of plots
marker_style_list = ["o", "d", "x"]
line_style_list   = ["solid", "dotted", "dashdot", (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 2, 1, 2, 1))]
model_labels_list = args.labels

lbl      = "Car"
lbl      = lbl.lower()

# Load the iou_vals
iou_vals = np.array(iou_vals)
roc_1    = []
roc_2    = []
for i, dis in enumerate(dis_keys):
    roc_1.append([m3d_res["res_{}m_{}".format(dis, iou)]["det_3d_" + lbl][1] for iou in iou_keys])
for j in range(num_input_folders):
    roc_2.append([])
    for i, dis in enumerate(dis_keys):
        roc_2[j].append([our_res[j]["res_{}m_{}".format(dis, iou)]["det_3d_" + lbl][1] for iou in iou_keys])


# Make a custom legend
patch1 = mpatches.Patch(color= colors_list[0], label= "[  0, 15]m box")
patch2 = mpatches.Patch(color= colors_list[1], label= "(15, 30]m box")
patch3 = mpatches.Patch(color= colors_list[3], label= "(45, 60]m box")
patch6 = mpatches.Patch(color= colors_list[2], label= "(30, 45]m box")

patch4 = Line2D([0], [0], color= "black", lw= fig_lw, label= label_folder1, linestyle= "--")
legend_handles = [patch4]
for j in range(num_input_folders-1, -1, -1):
    patch5 = Line2D([0], [0], color= "black", lw= fig_lw, label= model_labels_list[j], linestyle= line_style_list[j])
    legend_handles.append(patch5)
legend_handles.append(patch1)
legend_handles.append(patch2)
legend_handles.append(patch6)
legend_handles.append(patch3)
# =======================================
# Plot the usual plot
# =======================================
plt.figure(figsize= fig_size, dpi= params.DPI)
for i in range(len(dis_keys)):
    c = colors_list[i]
    plt.plot(iou_vals, roc_1[i], color=c, lw= fig_lw, linestyle="--")
    for j in range(num_input_folders):
        plt.plot(iou_vals, roc_2[j][i], color=c, lw= fig_lw, linestyle= line_style_list[j])

plt.xlabel(r"IoU$_{3D}$ Threshold")
plt.ylabel(r"AP$_{3D}$")
plt.xlim(np.min(iou_vals), np.max(iou_vals))
plt.ylim(bottom= bottom_ylim, top= top_ylim)
plt.grid()
plt.legend(handles= legend_handles, loc= "lower left", fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
plt.subplots_adjust(left= 0.110, right= 0.955, top= 0.960, bottom= 0.130)

save_path = os.path.join("images", "AP3D_vs_IOU3D_threshold_at_different_gt_distances_" + lbl + ".png")
savefig(plt, save_path, tight_flag= True, pad_inches= pad_inches)
plt.close()

# =======================================
# Plot the semi-log plot
# =======================================
fig = plt.figure(figsize= fig_size, dpi= params.DPI)
for i in range(len(dis_keys)):
    c = colors_list[i]
    plt.plot(iou_vals, roc_1[i], color=c, lw= fig_lw, linestyle="--")
    for j in range(num_input_folders):
        plt.semilogy(iou_vals, roc_2[j][i], color=c, lw= fig_lw, linestyle= line_style_list[j])

plt.xlabel(r"IoU$_{3D}$ Threshold")
plt.ylabel(r"log AP$_{3D}$")
plt.xlim(np.min(iou_vals), np.max(iou_vals))
plt.ylim(top= top_ylim)
plt.grid()
plt.legend(handles= legend_handles, loc= "lower left", fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
plt.subplots_adjust(left= 0.110, right= 0.955, top= 0.960, bottom= 0.130)

save_path = os.path.join("images", "AP3D_log_vs_IOU3D_threshold_at_different_gt_distances_" + lbl + ".png")
savefig(plt, save_path, tight_flag= True, pad_inches= pad_inches)
plt.close()