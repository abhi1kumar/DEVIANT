"""
    Sample Run:
    python plot/plot_sesn_basis.py
"""
import os, sys
from matplotlib import cm

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

from lib.projective.ses_conv import *


def get_basis(scales, effective_size):
    kernel_size = effective_size + 2
    padding     = kernel_size//2
    m1 = nn.Sequential(SESConv_Z2_H(in_channels= in_channel, out_channels= out_channel, kernel_size= kernel_size, effective_size= effective_size,
                                                                scales= scales, stride= 1, padding= padding, dilation= dilation))

    basis1 = m1[0].basis
    basis1_np = basis1.cpu().float().numpy()

    return m1, basis1, basis1_np

cmap        = 'rainbow'#'gist_rainbow'#'magma'#'plasma'#'viridis'#'seismic'
in_channel  = 64
out_channel = 64
show_all_scales = True
dilation    = 1
max_filters = 10
fs = 12
matplotlib.rcParams.update({'font.size': fs})

scale_sets = [[0.83, 0.9, 1.0]] #[[0.9, 1.26, 1.76]]#[[0.83, 1.0, 1.2], [0.9, 1.26, 1.76], [0.5, 1.0, 1.5]]
#===================================================================
# Effective size 3
#===================================================================

effective_size = 3
m1, basis1, basis1_np = get_basis(scale_sets[0], effective_size= effective_size)

num_scale_sets = len(scale_sets)
vmax = 0.8#np.max(basis1_np)
vmin = np.min([np.min(basis1_np), -vmax])
print(np.min(basis1_np), np.max(basis1_np), vmin, vmax)

if show_all_scales:
    num_scales = basis1.shape[1]
else:
    num_scales = 1

num_filters = basis1.shape[0]
if num_filters > max_filters:
    num_filters = max_filters

print("Showing for {} scale sets each one with scales x filters= {} x {}".format(num_scale_sets, num_scales, num_filters))
fig = plt.figure(figsize= (params.size[0]//2,  params.size[1]//2), dpi= params.DPI*3)

for k in range(num_scale_sets):
    for i in range(num_scales):
        for j in range(num_filters):
            scale_curr = i
            plt.subplot(num_scales*num_scale_sets,num_filters, k*num_filters*num_scales + i*num_filters + j +1)

            if k == 0:
                plt.imshow(basis1_np[j][scale_curr], vmin= vmin, vmax= vmax, cmap= cmap)

            plt.axis('off')
            if j == 0:
                title_text = ",".join(str(x) for x in scale_sets[k])
                txt_val = 'Scale ' + str(i)
                plt.text(-9, 3.0, txt_val, fontsize= fs)

fig.tight_layout(h_pad= -5.6, w_pad= 0.1)

# left, bottom, width, height
cax = plt.axes([0.98, 0.3, 0.03, 0.42])
cbar = plt.colorbar(cax=cax)
cbar_ticks = np.array([vmin,vmin/2,0, vmax, vmax/2])
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(cbar_ticks)

save_path = "images/sesn_basis_eff_size_" + str(effective_size) + ".png"
savefig(plt, save_path)
# plt.show()


#===================================================================
# Effective size 7
#===================================================================
effective_size = 7

max_filters = 8

# cmap        = 'seismic'#'rainbow'#'seismic'#'bwr'#'rainbow'
cmap = diverge_map(params.color_set1_pink/255.0, params.color_set1_cyan/255.0)
# cmap = diverge_map(params.color_set2_pink/255.0, params.color_set2_cyan/255.0)
# cmap = diverge_map(params.color_red/255.0, params.color_set2_cyan/255.0)
# cmap = diverge_map(params.color_red/255.0, np.array([30, 144, 255])/255.0)
# cmap = diverge_map(np.array([30, 144, 255])/255.0, params.color_red/255.0)
m1, basis1, basis1_np = get_basis(scale_sets[0], effective_size= effective_size)

num_scale_sets = len(scale_sets)
vmax = 0.8#np.max(basis1_np)
vmin = -vmax#np.min([np.min(basis1_np), -vmax])
print(np.min(basis1_np), np.max(basis1_np), vmin, vmax)

if show_all_scales:
    num_scales = basis1.shape[1]
else:
    num_scales = 1

num_filters = basis1.shape[0]
if num_filters > max_filters:
    num_filters = max_filters

print("Showing for {} scale sets each one with scales x filters= {} x {}".format(num_scale_sets, num_scales, num_filters))
fig = plt.figure(figsize= (params.size[0]//2,  params.size[1]//2), dpi= params.DPI*3)

for k in range(num_scale_sets):
    for i in range(num_scales):
        for j in range(num_filters):
            scale_curr = i
            plt.subplot(num_scales*num_scale_sets,num_filters, k*num_filters*num_scales + i*num_filters + j +1)

            if k == 0:
                plt.imshow(basis1_np[j][scale_curr], vmin= vmin, vmax= vmax, cmap= cmap)

            # plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            if j == 0:
                title_text = ",".join(str(x) for x in scale_sets[k])
                txt_val = 'Scale ' + str(i)
                plt.text(-13, 5.5, txt_val, fontsize= fs)


fig.tight_layout(h_pad= -5.2, w_pad= 0.1)

# left, bottom, width, height
cax = plt.axes([0.98, 0.26, 0.03, 0.48])
cbar = plt.colorbar(cax=cax)
cbar_ticks = np.array([vmin,vmin/2,0, vmax/2, vmax])
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([])

for j, lab in enumerate(cbar_ticks):
    cbar.ax.text(3.5, j/ 2.5 - 0.8, lab, ha='center', va='center')
# cbar.set_ticklabels(cbar_ticks)

save_path = "images/sesn_basis_eff_size_" + str(effective_size) + ".png"
savefig(plt, save_path)
