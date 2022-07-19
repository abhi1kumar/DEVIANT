"""
    Sample Run:
    python test/test_ses_equivariance.py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn

np.set_printoptions   (precision= 4, suppress= True)
torch.set_printoptions(precision= 4)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib
import cv2
import torch.nn.functional as F

import matplotlib.patches as patches
from lib.projective.ses_conv import *
from lib.projective.projective_utils import LogPolarConvolution
from lib.helpers.file_io import imread

def forward_over_model(img, model, max_over_scale= True):
    output = model(img)

    # print("Output shape= {} ".format(output.shape))
    output_np = output.detach().numpy()[0,0]
    # output_np = (output_np * 255).astype(np.uint8)
    if max_over_scale:
        output_np_max = np.max(output_np, axis= 0)
        index_max     = np.argmax(output_np, axis=0)
    else:
        output_np_max = output_np
        index_max = None

    return output_np, output_np_max, index_max

def convert_to_tensor(input_np, normalize= True):
    output = torch.from_numpy(input_np).unsqueeze(0).unsqueeze(0).float()

    if normalize:
        output /= 255.0

    return output

def get_pair_constraint_status(output_big, output_small, margin= 0.0):
    EPS = 1e-6

    if (output_big - output_small >= margin or torch.abs(output_big) < EPS or torch.abs(output_small) < EPS):
        return 1
    else:
        return 0

def constraint_satisfied(response_small, response_mid, response_big, margin= 0.0):
    """Checks if all the constraints are satisfied"""

    check_arr = np.zeros((6, )).astype(np.uint8)

    check_arr[0] = get_pair_constraint_status(output_big= response_small[0], output_small= response_small[1], margin= margin)
    check_arr[1] = get_pair_constraint_status(output_big= response_small[0], output_small= response_small[2], margin= margin)

    check_arr[2] = get_pair_constraint_status(output_big= response_mid[1], output_small= response_mid[0], margin= margin)
    check_arr[3] = get_pair_constraint_status(output_big= response_mid[1], output_small= response_mid[2], margin= margin)

    check_arr[5] = get_pair_constraint_status(output_big= response_big[2], output_small= response_big[1], margin= margin)
    check_arr[4] = get_pair_constraint_status(output_big= response_big[2], output_small= response_big[0], margin= margin)

    check_sum = np.sum(check_arr)
    if check_sum == 6:
        return True, check_sum
    else:
        return False, check_sum


def get_response_of_filter_bank(basis, show_optimization= True, opt_margin= 0.0, check_margin= 0.0):

    if basis.shape[2] == 5:
        response_small_all = basis[:,:,2,2]
        response_mid_all   = torch.sum(torch.sum(basis[:,:,1:4,1:4], dim= 3), dim= 2)
        response_big_all   = torch.sum(torch.sum(basis[:,:,:,:], dim= 3), dim= 2)
    elif basis.shape[2] == 9:
        response_small_all = basis[:,:,4,4]
        response_mid_all   = torch.sum(torch.sum(basis[:,:,2:7,2:7], dim= 3), dim= 2)
        response_big_all   = torch.sum(torch.sum(basis[:,:,:,:], dim= 3), dim= 2)


    num_filters = basis.shape[0]
    num_satisfied_constraints = 0

    for filter_ind in range(num_filters):
        response_small = response_small_all[filter_ind]
        response_mid   = response_mid_all  [filter_ind]
        response_big   = response_big_all  [filter_ind]

        # Check if all constraints are satisfied
        check_flag, check_sum = constraint_satisfied(response_small, response_mid, response_big, margin= check_margin)
        num_satisfied_constraints += check_sum
        if check_flag:
            pass
            # print("Filter {} satisfied \o/".format(filter_ind, check_sum))
            # print(response_small, response_mid, response_big)
        else:
            pass
            # Find smaller numbers to be multiplied to the filters
            # print("Filter {} NOT satisfied with only {} satisfied".format(filter_ind, check_sum))
            # print(response_small, response_mid, response_big)

    return int(num_satisfied_constraints)

effective_size = 3
get_toy_gif          = False
use_global_vmax_vmin = False
skip_first_filter    = False

write_to_gif   = True
cmap           = 'jet'
vmax_global    = 1
vmin_global    = -vmax_global
vmin_corr   = -1
vmin_corr_toy = -0.34
vmax_corr_toy = 4.0
show_log_polar= False
sesn_scales = [0.83, 1.0, 1.2]

kernel_size = effective_size + 2
padding     = 0 #kernel_size//2
stride      = kernel_size//2
filter_set_index = 0
width       = 32
height      = 32

if show_log_polar:
    num_rows    = 2
    num_cols    = 10
else:
    num_rows    = 4
    num_cols    = 5


num_func  = effective_size * effective_size
num_scales= len(sesn_scales)
toy_image = np.zeros((num_scales, kernel_size, kernel_size))
mask      = np.zeros(toy_image.shape).astype(np.uint8)
if effective_size == 3:
    mask[0, 2, 2] = 1
    mask[1, 1:4, 1:4] = 1
    mask[2] = 1
elif effective_size == 7:
    mask[0, 4, 4] = 1
    mask[1, 2:7, 2:7] = 1
    mask[2] = 1
toy_image[mask > 0] = 1
toy_image_all = toy_image.reshape((num_scales, -1))

fig_sim, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize= (params.size[0]*1.5, params.size[1]*2), dpi= params.DPI)

for rescale_index, rescale_basis in enumerate([False, True]):
    for norm_index, norm_per_scale in enumerate([False, True]):

        gif_path_prefix = "images/test_ses_basis_orthogonality_size_" + str(effective_size)
        gif_path = gif_path_prefix
        if get_toy_gif:
            gif_path += "_toy"
        if norm_per_scale:
            print("Normalizing each filter by its norm")
            gif_path += "_norm"
        if rescale_index:
            print("Rescaling the filter to match toy example")
            gif_path += "_rescale"
        gif_path    += ".gif"

        if effective_size == 7:
            model_ses = nn.Sequential(SESConv_Z2_H(in_channels= 1, out_channels= 1, kernel_size= kernel_size, effective_size=effective_size,
                                                   scales= sesn_scales, stride= stride, padding= padding, bias= False, padding_mode= 'constant', norm_per_scale= norm_per_scale, dilation= 1
                                                   , rescale_basis= rescale_basis
                                                   ))
            ses_layer_index = 0
        else:
            model_ses = nn.Sequential(SESCopy_Scale(scales= sesn_scales), SESConv_H_H(in_channels= 1, out_channels= 1, kernel_size= kernel_size, effective_size=effective_size,
                                                   scales= sesn_scales, stride= stride, padding= padding, bias= False, padding_mode= 'constant', norm_per_scale= norm_per_scale, dilation= 1,
                                                   scale_size= 1, rescale_basis= rescale_basis))
            ses_layer_index = 1

        model_logpolar = nn.Sequential ( LogPolarConvolution (nn.Conv2d(in_channels= 1, out_channels= 1, kernel_size= effective_size,
                                                                        stride= stride, padding= padding, bias= False, dilation= 1)      ))

        print(model_ses[ses_layer_index].weight.shape)
        print(model_ses[ses_layer_index].basis.shape)

        num_filter_sets     = model_ses[ses_layer_index].basis.shape[0]
        if write_to_gif:
            gif_writer          = open_gif_writer(file_path= gif_path, duration= 1.5)

        if skip_first_filter:
            start_index = 1
        else:
            start_index = 0

        for filter_set_index in range(start_index, num_filter_sets):
            if get_toy_gif:
                img_single_ch_np         = toy_image[0]
                img_scaled_single_ch_np  = toy_image[1]
                if len(sesn_scales) > 2:
                    img_scaled_single_ch_np2 = toy_image[2]
            else:
                img_single_ch_np         = model_ses[ses_layer_index].basis[filter_set_index, 0].detach().numpy()
                img_scaled_single_ch_np  = model_ses[ses_layer_index].basis[filter_set_index, 1].detach().numpy()
                if len(sesn_scales) > 2:
                    img_scaled_single_ch_np2 = model_ses[ses_layer_index].basis[filter_set_index, 2].detach().numpy()
    
            normalize  = False
    
            # Convert them to tensors
            img_tensor        = convert_to_tensor(img_single_ch_np       , normalize= normalize)
            img_scaled_tensor = convert_to_tensor(img_scaled_single_ch_np, normalize= normalize)
            if len(sesn_scales) > 2:
                img_scaled_tensor2 = convert_to_tensor(img_scaled_single_ch_np2, normalize= normalize)
    
            if ses_layer_index == 0:
                our_weight = torch.zeros((1, 1, effective_size * effective_size ) )
                our_weight[0, 0, filter_set_index] = 1
            else:
                our_weight = torch.zeros((1, 1, 1, effective_size * effective_size ) )
                our_weight[0, 0, 0, filter_set_index] = 1
            # Make sure you fire the correct filter
            model_ses[ses_layer_index].weight = torch.nn.Parameter(our_weight)
    
            output_np, output_np_max, index_max = forward_over_model(img_tensor, model_ses)
            output_scaled_np, output_scaled_np_max, index_scaled_max = forward_over_model(img_scaled_tensor, model_ses)
            if len(sesn_scales) > 2:
                output_scaled_np2, output_scaled_np_max2, index_scaled_max2 = forward_over_model(img_scaled_tensor2, model_ses)
            # print("Max out image1 = " , np.max(output_np_max[output_np_max > 0]))
            # print("Max out image1 = " , np.max(output_scaled_np_max[output_scaled_np_max > 0]))


            if use_global_vmax_vmin:
                # Use toy vmax vmin for all filters
                if skip_first_filter:
                    vmax = vmax_global
                    vmin = vmin_global
                else:
                    vmax = vmax_corr_toy
                    vmin = vmin_corr_toy
            else:
                if len(sesn_scales) > 2:
                    vmax = np.max([np.max(output_np_max), np.max(output_scaled_np_max), np.max(output_scaled_np_max2), np.max(img_single_ch_np), np.max(img_scaled_single_ch_np), np.max(img_scaled_single_ch_np2)])
                    vmin = np.min([np.min(output_np_max), np.min(output_scaled_np_max), np.min(output_scaled_np_max2), np.min(img_single_ch_np), np.min(img_scaled_single_ch_np), np.min(img_scaled_single_ch_np2)])
                else:
                    vmax = np.max([np.max(output_np_max), np.max(output_scaled_np_max), np.max(img_single_ch_np), np.max(img_scaled_single_ch_np)])
                    vmin = np.min([np.min(output_np_max), np.min(output_scaled_np_max), np.min(img_single_ch_np), np.min(img_scaled_single_ch_np)])
    
            print("Filter Index={} vmin= {:.2f} vmax= {:.2f}".format(filter_set_index, vmin, vmax))
    
            fig = plt.figure(figsize= (params.size[0]*1.5, params.size[1]*1.75), dpi= params.DPI)
            plt.subplot(num_rows, num_cols, 1)
            plt.title('Filter Set ' + str(filter_set_index))
            plt.axis('off')
    
            plt.subplot(num_rows, num_cols, 2)
            plt.imshow(model_ses[ses_layer_index].basis[filter_set_index, 0].detach().numpy(), vmin= vmin, vmax= vmax)
            plt.title('Scale 0')
            plt.axis('off')
            plt.subplot(num_rows, num_cols, 3)
            plt.imshow(model_ses[ses_layer_index].basis[filter_set_index, 1].detach().numpy(), vmin= vmin, vmax= vmax)
            plt.title('Scale 1')
            plt.axis('off')
            if len(sesn_scales) > 2:
                plt.subplot(num_rows, num_cols, 4)
                plt.imshow(model_ses[ses_layer_index].basis[filter_set_index, 2].detach().numpy(), vmin= vmin, vmax= vmax)
                plt.title('Scale 2')
                plt.axis('off')
    
            # Input0 plots
            plt.subplot(num_rows,num_cols,6)
            plt.imshow(img_single_ch_np, vmin= vmin, vmax= vmax)
            plt.title('I/P 0')
            plt.axis('off')
            sub = plt.subplot(num_rows, num_cols,7)
            plt.imshow(output_np[0], vmin= vmin, vmax= vmax)
            plt.axis('off')
            plt.subplot(num_rows, num_cols,8)
            plt.imshow(output_np[1], vmin= vmin, vmax= vmax)
            plt.axis('off')
            plt.subplot(num_rows, num_cols,9)
            if len(sesn_scales) > 2:
                plt.imshow(output_np[2], vmin= vmin, vmax= vmax)
            plt.axis('off')
            # sub = plt.subplot(num_rows, num_cols,10)
            # plt.imshow(output_np_max, vmin= vmin, vmax= vmax)
            # plt.title('Max')
            # plt.axis('off')
            if len(index_max) == 1 and index_max.flatten() != 0:
                draw_red_border(sub)
    
            # Input1 plots
            plt.subplot(num_rows, num_cols,11)
            plt.imshow(img_scaled_single_ch_np, vmin= vmin, vmax= vmax)
            plt.title('I/P 1')
            plt.axis('off')
            plt.subplot(num_rows, num_cols,12)
            plt.imshow(output_scaled_np[0], vmin= vmin, vmax= vmax)
            plt.axis('off')
            sub1 = plt.subplot(num_rows, num_cols,13)
            plt.imshow(output_scaled_np[1], vmin= vmin, vmax= vmax)
            plt.axis('off')
            plt.subplot(num_rows, num_cols,14)
            if len(sesn_scales) > 2:
                plt.imshow(output_scaled_np[2], vmin= vmin, vmax= vmax)
            plt.axis('off')
            # sub1 = plt.subplot(num_rows, num_cols,15)
            # plt.imshow(output_scaled_np_max, vmin= vmin, vmax= vmax)
            # plt.title('Max')
            # plt.axis('off')
            if len(index_scaled_max) == 1 and index_scaled_max.flatten() != 1:
                draw_red_border(sub1)
    
            # Input2 plots
            if len(sesn_scales) > 2:
                plt.subplot(num_rows, num_cols,16)
                plt.imshow(img_scaled_single_ch_np2, vmin= vmin, vmax= vmax)
                plt.title('I/P 2')
                # plt.axis('off')
                plt.subplot(num_rows, num_cols,17)
                plt.imshow(output_scaled_np2[0], vmin= vmin, vmax= vmax)
                # plt.axis('off')
                plt.subplot(num_rows, num_cols,18)
                plt.imshow(output_scaled_np2[1], vmin= vmin, vmax= vmax)
                plt.axis('off')
                sub2 = plt.subplot(num_rows, num_cols,19)
                plt.imshow(output_scaled_np2[2], vmin= vmin, vmax= vmax)
                plt.axis('off')
                # sub2 = plt.subplot(num_rows, num_cols,20)
                # plt.imshow(output_scaled_np_max2, vmin= vmin, vmax= vmax)
                # plt.title('Max')
                # plt.axis('off')
                if len(index_scaled_max2) == 1 and index_scaled_max2.flatten() != 2:
                    draw_red_border(sub2)
    
            if show_log_polar:
                output_np, output_np_max               = forward_over_model(img_tensor, model_logpolar, max_over_scale= False)
                output_scaled_np, output_scaled_np_max = forward_over_model(img_scaled_tensor, model_logpolar, max_over_scale= False)
    
                plt.subplot(num_rows,num_cols,7)
                plt.imshow(img_single_ch_np, vmin= vmin, vmax= vmax)
                plt.axis('off')
                # plt.subplot(num_rows, num_cols,8)
                # plt.imshow(output_np[0], vmin= vmin, vmax= vmax)
                plt.subplot(num_rows, num_cols,9)
                plt.imshow(output_np_max, vmin= vmin, vmax= vmax)
                plt.axis('off')
    
                plt.subplot(num_rows, num_cols,10)
                plt.imshow(img_scaled_single_ch_np, vmin= vmin, vmax= vmax)
                plt.axis('off')
                plt.subplot(num_rows, num_cols,12)
                plt.imshow(output_scaled_np_max, vmin= vmin, vmax= vmax)
                plt.axis('off')
    
            if write_to_gif:
                ubyte_image = convert_fig_to_ubyte_image(fig)
                add_ubyte_image_to_gif_writer(gif_writer, ubyte_image)
            else:
                plt.show()
            plt.close()
    
        if write_to_gif:
            close_gif_writer(gif_writer)

        # Plot the big similarity matrix of filters
        basis_all = model_ses[ses_layer_index].basis.cpu().numpy() # num_func x scale x h x w
        num_func, num_scales, h, w = basis_all.shape
        basis_all_reshape = basis_all.reshape((num_func * num_scales, h * w))

        similarity = np.tensordot(basis_all_reshape, basis_all_reshape, axes=(1, 1))

        # ax = plt.gca()
        # Plot all the correlation responses
        if rescale_index == 0:
            if norm_index == 0:
                axp = ax1.imshow(similarity, vmin= vmin_corr, vmax= 1, cmap= cmap)
                ax1.set_title('Vanilla')
            else:
                axp = ax2.imshow(similarity, vmin= vmin_corr, vmax= 1, cmap= cmap)
                ax2.set_title('Normalize')
        else:
            if norm_index == 0:
                axp = ax3.imshow(similarity, vmin= vmin_corr, vmax= 1, cmap= cmap)
                ax3.set_title('Rescale')
            else:
                axp = ax4.imshow(similarity, vmin= vmin_corr, vmax= 1, cmap= cmap)
                ax4.set_title('Normalize + Rescale')
                cbar_ax = fig_sim.add_axes([0.92, 0.15, 0.03, 0.7])
                fig_sim.colorbar(axp, cax=cbar_ax)


        # Plot responses with toy example
        # basis_all_permute  = basis_all.transpose(1, 0, 2, 3) # scale x num_func x h x w
        basis_all_reshape2 = basis_all.reshape((num_func * num_scales, h * w))

        similarity2 = np.tensordot(basis_all_reshape2, toy_image_all, axes=(1, 1))

        # Get how many constraints satisfied
        basis_tensor = model_ses[ses_layer_index].basis
        num_satisfied_constraints = get_response_of_filter_bank(basis= basis_tensor, check_margin= 0.01)
        print("==> Number of constraints satisfied = {:3d}/{:3d}".format(num_satisfied_constraints, basis_tensor.shape[0]*6))

        """
        # Plot toy responses
        if rescale_index == 0:
            if norm_index == 0:
                axp = ax5.imshow(similarity2, vmin= vmin_corr_toy, vmax= vmax_corr_toy, cmap= cmap)
                ax5.set_title('Vanilla')
            else:
                axp = ax6.imshow(similarity2, vmin= vmin_corr_toy, vmax= vmax_corr_toy, cmap= cmap)
                ax6.set_title('Normalize')
        else:
            if norm_index == 0:
                axp = ax7.imshow(similarity2, vmin= vmin_corr_toy, vmax= vmax_corr_toy, cmap= cmap)
                ax7.set_title('Rescale')
            else:
                axp = ax8.imshow(similarity2, vmin= vmin_corr_toy, vmax= vmax_corr_toy, cmap= cmap)
                ax8.set_title('Normalize + Rescale')
        """
        print("")

img_path = "images/test_ses_equivariance_" + str(effective_size) + "_similarity.png"
savefig(plt, path= img_path)
# plt.show()
