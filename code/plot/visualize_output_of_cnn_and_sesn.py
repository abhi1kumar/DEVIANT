"""
    Visualizes output of CNN and SESN and shows that SESN obey scale equivariance property
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn

np.set_printoptions   (precision= 4, suppress= True)
torch.set_printoptions(precision= 2, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

from lib.projective.ses_conv import *
from lib.helpers.util import *
from matplotlib import pyplot as plt
from plot.common_operations import *

basis_index = 0
kernel_size = 5
padding     = 0#kernel_size//2
sesn_scales = [0.7, 0.83, 1.0]
cmap        = 'magma'
cmap_min    = 0#np.min(feature_img)
circ_size   = 1
circ_rad    = 'r'
pause_time  = 0.002
plot_gif    = False
gif_time    = 1.0
show_downsampled = True
if show_downsampled:
    matplotlib.rcParams.update({'font.size': 12})
else:
    matplotlib.rcParams.update({'font.size': 18})

scale_list = 1.0/np.linspace(1.0, 1.7, 10)
# scale_list = 1.0/np.array([1.0, 1.2, 1.44])

# Bare minimum scales to visualize the output
sesn_scales = [0.5, 1.0]
scale_list = np.array([sesn_scales[0]])


effective_size = kernel_size - 2
Hin = 4*kernel_size - 1
Win = Hin

model = nn.Sequential(SESConv_Z2_H(in_channels=1, out_channels= 1, kernel_size= kernel_size, effective_size= effective_size, padding= padding, stride= 1, scales= sesn_scales))
ses_layer_index_in_model = 0
# We make a second model with different sesn_scales and see if they give the same output.
model2 = nn.Sequential(SESConv_Z2_H(in_channels=1, out_channels= 1, kernel_size= kernel_size, effective_size= effective_size, padding= padding, stride= 1, scales= [0.83, 0.9, 1.0]))
print(torch.norm(model[0].basis[basis_index, 1] - model2[0].basis[basis_index, 1]))
# print(torch.norm(model[0].basis[basis_index, 2] - model2[0].basis[basis_index, 1]))
print(torch.norm(model2[0].basis[basis_index, 0] - model2[0].basis[basis_index, 1]))
print(torch.norm(model2[0].basis[basis_index, 1] - model2[0].basis[basis_index, 2]))
# print(model[0].basis[basis_index, 1])
# print(model2[0].basis[basis_index, 0])
# print(model2[0].basis[basis_index, 1])
# print(model2[0].basis[basis_index, 2])
# sys.exit(0)



# model = nn.Sequential(SESCopy_Scale(scales= sesn_scales), SESConv_H_H(in_channels=1, out_channels= 1,\
#                                       kernel_size= kernel_size, effective_size= effective_size, scale_size= 1, padding= 0, stride= 1, scales= sesn_scales))
# ses_layer_index_in_model = 1


ses_layer_weight = torch.zeros((1, 1, effective_size*effective_size))
ses_layer_weight[0, 0, basis_index] = 1.0
model[ses_layer_index_in_model].weight  = torch.nn.Parameter(ses_layer_weight)

img_numpy   = np.ones((Hin, Win, 3)).astype(np.uint8)*255
img_float   = torch.Tensor(img_numpy.transpose(2, 0, 1)[0][np.newaxis, ] /255.0).unsqueeze(0)
print("Image shape= ", img_float.shape)
feature_img = model(img_float)[0, 0].detach().cpu().float().numpy()

if not show_downsampled:
    num_rows    = 2
    shift_row   = 0
else:
    num_rows    = 3
    shift_row   = 4

cmap_max  = max(np.max(feature_img), 1.0)

if plot_gif:
    gif_file_path = "images/ses_in_cont_domain_kernel_size_" + str(kernel_size) + "_basis_" + str(basis_index) + ".gif"
    gif_writer    = open_gif_writer(gif_file_path, duration= gif_time)
fig           = plt.figure(dpi= params.DPI)

for i, scale_curr in enumerate(scale_list):
    img_np_new    = get_downsampled_image(img= img_numpy, curr_scale= 1.0/scale_curr)
    img_float_new = torch.Tensor(img_np_new.transpose(2, 0, 1)[0][np.newaxis, ] / 255.0).unsqueeze(0)

    feature_new   = model(img_float_new)[0, 0].detach().cpu().float().numpy()

    plt.subplot(num_rows, 4, 1)
    plt.imshow(cmap_max*img_numpy[:, :, 0]/255.0, cmap= cmap, vmin= cmap_min, vmax= cmap_max)
    plt.title('Image ' +r'$(h)$')
    plt.axis('off')

    plt.subplot(num_rows, 4, 5 + shift_row)
    plt.imshow(cmap_max*img_np_new[:, :, 0]/255.0, cmap= cmap, vmin= cmap_min, vmax= cmap_max)
    plt.title(r'$T_{' + "{:.1f}".format(scale_curr) + r'}(h)$')
    plt.axis('off')

    plt.subplot(num_rows, 4, 2)
    plt.imshow(feature_img[0], cmap= cmap, vmin= cmap_min, vmax= cmap_max)
    scale0_max = np.max(feature_img[0])
    # plt.title("{:.1f}".format(scale0_max))
    plt.title(r'$h*\Psi_{' + "{:.1f}".format(sesn_scales[0]) + r'}$')
    plt.axis('off')
    ind = np.unravel_index(feature_img[0].argmax(), feature_img[0].shape)
    circle1 = plt.Circle((ind[1], ind[0]), circ_size, color= circ_rad)
    # plt.gca().add_patch(circle1)

    plt.subplot(num_rows, 4, 3)
    plt.imshow(feature_img[1], cmap= cmap, vmin= cmap_min, vmax= cmap_max)
    scale1_max = np.max(feature_img[1])
    plt.title(r'$h*\Psi_{' + "{:.1f}".format(sesn_scales[1]) + r'}$')
    # plt.title("{:.1f}".format(scale1_max))
    plt.axis('off')
    ind = np.unravel_index(feature_img[1].argmax(), feature_img[1].shape)
    circle1 = plt.Circle((ind[1], ind[0]), circ_size, color= circ_rad)
    # plt.gca().add_patch(circle1)

    if feature_img.shape[0] > 2:
        plt.subplot(num_rows, 4, 4)
        plt.imshow(feature_img[2], cmap= cmap, vmin= cmap_min, vmax= cmap_max)
        scale2_max = np.max(feature_img[2])
        # plt.title("{:.1f}".format(scale2_max))
        plt.title(r'$h*\Psi_{' + "{:.1f}".format(sesn_scales[2]) + r'}$')
        plt.axis('off')
        ind = np.unravel_index(feature_img[2].argmax(), feature_img[2].shape)
        circle1 = plt.Circle((ind[1], ind[0]), circ_size, color= circ_rad)
        # plt.gca().add_patch(circle1)

    if show_downsampled:
        plt.subplot(num_rows, 4, 6)
        print(feature_img[0].shape)
        plt.imshow(get_downsampled_image(feature_img[0][np.newaxis].repeat(3, axis=0), curr_scale= 1.0/scale_curr, height_first_flag= False)[0],  cmap= cmap, vmin= cmap_min, vmax= cmap_max)
        plt.title(r'$T_{' + "{:.1f}".format(scale_curr) + '}(h*\Psi_{' + "{:.1f}".format(sesn_scales[0]) + r'})$')
        plt.axis('off')

        plt.subplot(num_rows, 4, 7)
        plt.imshow(get_downsampled_image(feature_img[1][np.newaxis].repeat(3, axis=0), curr_scale= 1.0/scale_curr, height_first_flag= False)[0], cmap= cmap, vmin= cmap_min, vmax= cmap_max)
        plt.title(r'$T_{' + "{:.1f}".format(scale_curr) + '}(h*\Psi_{' + "{:.1f}".format(sesn_scales[1]) + r'})$')
        plt.axis('off')

        if feature_img.shape[0] > 2:
            plt.subplot(num_rows, 4, 8)
            plt.imshow(get_downsampled_image(feature_img[2][np.newaxis].repeat(3, axis=0), curr_scale= 1.0/scale_curr, height_first_flag= False)[0], cmap= cmap, vmin= cmap_min, vmax= cmap_max)
            plt.title(r'$T_{' + "{:.1f}".format(scale_curr) + '}(h*\Psi_{' + "{:.1f}".format(sesn_scales[2]) + r'})$')
            plt.axis('off')


    plt.subplot(num_rows, 4, 6 + shift_row)
    plt.imshow(feature_new[0], cmap= cmap, vmin= cmap_min, vmax= cmap_max)
    scale0_max = np.max(feature_new[0])
    plt.title(r'$T_{' + "{:.1f}".format(scale_curr) + r'}(h)$' + r'$* \Psi_{' + "{:.1f}".format(sesn_scales[0]) + r'}$')
    plt.axis('off')
    ind = np.unravel_index(feature_new[0].argmax(), feature_new[0].shape)
    circle1 = plt.Circle((ind[1], ind[0]), circ_size, color= circ_rad)
    # plt.gca().add_patch(circle1)

    plt.subplot(num_rows, 4, 7 + shift_row)
    im = plt.imshow(feature_new[1], cmap= cmap, vmin= cmap_min, vmax= cmap_max)
    scale1_max = np.max(feature_new[1])
    # plt.title("{:.1f}".format(scale1_max))
    plt.title(r'$T_{' + "{:.1f}".format(scale_curr) + r'}(h)$' + r'$* \Psi_{' + "{:.1f}".format(sesn_scales[1]) + r'}$')
    plt.axis('off')
    ind = np.unravel_index(feature_new[1].argmax(), feature_new[1].shape)
    circle1 = plt.Circle((ind[1], ind[0]), circ_size, color= circ_rad)
    # plt.gca().add_patch(circle1)

    if feature_new.shape[0] > 2:
        plt.subplot(num_rows, 4, 8 + shift_row)
        plt.imshow(feature_new[2], cmap= cmap, vmin= cmap_min, vmax= cmap_max)
        scale2_max = np.max(feature_new[2])
        # plt.title("{:.1f}".format(scale2_max))
        plt.title(r'$T_{' + "{:.1f}".format(scale_curr) + r'}(h)$' + r'$* \Psi_{' + "{:.1f}".format(sesn_scales[2]) + r'}$')
        plt.axis('off')
        ind = np.unravel_index(feature_new[2].argmax(), feature_new[2].shape)
        circle1 = plt.Circle((ind[1], ind[0]), circ_size, color= circ_rad)
        # plt.gca().add_patch(circle1)

    if feature_img.shape[0] > 2:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.035, 0.7])
    else:
        cbar_ax = fig.add_axes([0.72, 0.15, 0.035, 0.7])
    fig.colorbar(im, cax=cbar_ax)


    if plot_gif:
        ubyte_image = convert_fig_to_ubyte_image(fig)
        add_ubyte_image_to_gif_writer(gif_writer, ubyte_image)

        if i != len(scale_list)-1:
            # plt.close()
            plt.draw()
            plt.pause(pause_time)
        plt.clf()
    else:
        if show_downsampled:
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
        # plt.tight_layout()
        savefig(plt, "images/visualize_output_of_cnn_sesn.png")
        plt.show()

if plot_gif:
    close_gif_writer(gif_writer)