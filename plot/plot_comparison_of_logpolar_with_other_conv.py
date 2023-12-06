"""
    Sample Run:
    python plot/plot_comparison_of_logpolar_with_other_conv.py
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

import logging
logging.basicConfig(filename="test/log.txt", level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def get_filter_on_image(h, w, filter_type= "vanilla"):

    filter_on_image = np.zeros((h, w, 3)).astype(np.uint8)
    cnt     = 0
    color1  = params.color_blue#params.color_set1_cyan#np.array([30, 255, 255])
    color2  = params.color_set1_cyan#np.array([30, 144, 255])
    color5  = params.color_red#params.color_set1_pink#np.array([255, 30, 143])
    color6  = params.color_set1_pink_light#np.array([255, 143, 30])
    color3  = params.color_set1_yellow#np.array([143, 255, 30])
    color4  = params.color_yellow    #np.array([30 , 255, 30])

    color1  = params.color_set2_cyan
    color2  = params.color_set2_cyan_light
    color3  = params.color_set2_pink
    color4  = params.color_set2_pink_light
    color5  = params.color_set2_yellow
    color6  = params.color_set2_yellow_light

    for i in range(h//skip_h):
        curr_i     = i*skip_h
        curr_i_fin = curr_i+skip_h
        if curr_i_fin >= h:
            curr_i_fin = h

        if filter_type == "vanilla":
            # Our filter
            color_temp_1 = color1
            color_temp_2 = color2

        elif filter_type == "garrick":
            # Garrick filter
            if curr_i < h/3.0:
                color_temp_1 = color1
                color_temp_2 = color2
            elif curr_i >= 2*h/3.0:
                color_temp_1 = color5
                color_temp_2 = color6
            else:
                color_temp_1 = color3
                color_temp_2 = color4

        for j in range(w//skip_w):
            curr_j     = j     *skip_w
            curr_j_fin = curr_j+skip_w
            if curr_j_fin >= w:
                curr_j_fin = w

            if (cnt + i) % 2 == 0:
                filter_on_image[curr_i: curr_i_fin, curr_j: curr_j_fin] = color_temp_1
            else:
                filter_on_image[curr_i: curr_i_fin, curr_j: curr_j_fin] = color_temp_2
            cnt += 1

    return filter_on_image



#======================================================================
# Main starts here
#======================================================================
h         = 384
w         = 1280
skip_h    = 32
skip_w    = 32
w         = w - w%skip_w
h         = h - h%skip_h
(cX, cY)  = (w // 2, h // 2)


fs         = 20
matplotlib.rcParams.update({'font.size': fs})

img_path  = 'data/KITTI/training/image_2/000014.png'
im_pre    = imread(img_path)[:,:,::-1]

alpha     = 0.6
base_image= cv2.resize(im_pre, (w, h))

conv_on_image_conventional    = get_filter_on_image(h= h, w= w, filter_type="vanilla")
conv_on_image_ours            = cv2.logPolar(conv_on_image_conventional, center= (cX, cY), M= min(cX, cY), flags= cv2.WARP_INVERSE_MAP)
conv_on_image_garrick         = get_filter_on_image(h= h, w= w, filter_type="garrick")

# Convolutions on blank image
plt.figure(figsize= (params.size[0], params.size[1]*1.5), dpi= params.DPI)
plt.subplot(3,1,1)
plt.imshow(conv_on_image_conventional)
plt.axis('off')
plt.title('Vanilla Convolution')

plt.subplot(3,1,2)
plt.imshow(conv_on_image_garrick)
plt.axis('off')
plt.title('Depth-Aware Convolution [Brazil et al.]')

plt.subplot(3,1,3)
plt.imshow(conv_on_image_ours)
plt.axis('off')
plt.title('Log-polar Convolution')

savefig(plt, path="images/comparison_of_convolutions.png")
plt.show()
plt.close()

# Convolutions on image
plt.figure(figsize= (params.size[0], params.size[1]*1.5), dpi= params.DPI)
plt.subplot(3,1,1)
plt.imshow(conv_on_image_conventional)
plt.imshow(base_image, alpha= alpha)
plt.axis('off')
plt.title('Vanilla Convolution')

plt.subplot(3,1,2)
plt.imshow(conv_on_image_garrick)
plt.imshow(base_image, alpha= alpha)
plt.axis('off')
plt.title('Depth-Aware Convolution [Brazil et al.]')

plt.subplot(3,1,3)
plt.imshow(conv_on_image_ours)
plt.imshow(base_image, alpha= alpha)
plt.axis('off')
plt.title('Log-polar Convolution')

# savefig(plt, path="images/comparison_of_convolutions_with_image.png")