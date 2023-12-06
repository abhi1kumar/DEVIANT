

"""
    Stores the plotting settings to be used for plots.
    Import this module so that Tkinter or Agg can be checked

    Version 1 2020-05-25 Abhinav Kumar
"""
import matplotlib

# Check if Tkinter is there otherwise Agg
import imp
try:
    imp.find_module('_Tkinter')
    pass
except ImportError:
    matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np

DPI        = 200
ms         = 12
lw         = 5
alpha      = 0.9
size       = (10, 6)

# Remember to call update whenever you use a different fontsize
fs         = 28
matplotlib.rcParams.update({'font.size': fs})

# legend properties
# example
# plt.legend(handles= legend_handles, loc= "upper right", fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
legend_fs  = 16
legend_border_axes_plot_border_pad = 0.05
legend_border_pad                  = 0.1
legend_vertical_label_spacing      = 0.1
legend_marker_text_spacing         = 0.2

IMAGE_DIR  = "images"
dodge_blue =  np.array([0.12, 0.56, 1.0]) #np.array([30, 144, 255])/255.
color1     = (1,0.45,0.45)
color2     = "dodgerblue"

# basic ones
color_red    = np.array([255, 0, 0.0])
color_blue   = np.array([0, 0, 255.0])
color_yellow = np.array([255, 255, 0.0])

#triadic
color_set1_cyan   = np.array([59, 221, 255.0])
color_set1_pink   = np.array([255, 59, 141.0])
color_set1_yellow = np.array([255, 216, 59.0])

color_set1_cyan_light   = np.array([177, 225, 255.0])
color_set1_pink_light   = np.array([255, 177, 200.0])
color_set1_yellow_light = np.array([255, 243, 77.0])

num_bins = 50

color_set2_cyan        = np.array([0, 59, 157.0])
color_set2_cyan_light  = np.array([145,236, 255.0])
color_set2_yellow      = np.array([255, 180, 0.0])
color_set2_yellow_light= np.array([255, 255, 0.0])
color_set2_pink        = np.array([254, 0.0, 0.0])
color_set2_pink_light  = np.array([255, 213, 225.0])

display_frequency = 500
min_iou2d_overlap = 0.5
frac_to_keep      = 0.7
