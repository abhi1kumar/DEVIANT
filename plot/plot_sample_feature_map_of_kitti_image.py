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

import cv2

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib
from matplotlib import pyplot as plt

from lib.helpers.file_io import imread, imwrite
from lib.projective.ses_conv import *


def get_basis(effective_size):
    kernel_size = effective_size + 2
    padding     = kernel_size//2

    in_channel  = 1
    out_channel = 1
    dilation    = 1
    scales      = [0.83, 0.9, 1.0]

    m1 = nn.Sequential(SESConv_Z2_H(in_channels= in_channel, out_channels= out_channel, kernel_size= kernel_size, effective_size= effective_size,
                                                                scales= scales, stride= 1, padding= padding, dilation= dilation))

    basis1 = m1[0].basis
    basis1_np = basis1.cpu().float().numpy()

    return m1, basis1, basis1_np

imagepath = "data/KITTI/training/image_2/000008.png"
effective_size = 7

model, basis1, basis1_np = get_basis(effective_size= effective_size)


vmax = 0.8#np.max(basis1_np)
vmin = -vmax#np.min([np.min(basis1_np), -vmax])
cmap = diverge_map(params.color_set1_pink/255.0, params.color_set1_cyan/255.0)

img_numpy   = cv2.resize(imread(imagepath), (1280, 384), interpolation= cv2.INTER_AREA)/255.0
img_float   = torch.Tensor(img_numpy.transpose(2, 0, 1)[0][np.newaxis][np.newaxis])
print("Image shape= ", img_float.shape)

for k in range(1,2):
    basis_index = k
    ses_layer_weight = torch.zeros((1, 1, effective_size*effective_size))
    ses_layer_weight[0, 0, basis_index] = 1.0
    model[0].weight  = torch.nn.Parameter(ses_layer_weight)
    feature_img = model(img_float)[0, 0, 2].detach().cpu().float().numpy()

    # now plot
    plt.figure(figsize= params.size, dpi= params.DPI)
    plt.imshow(feature_img, vmin= vmin, vmax= vmax, cmap= cmap)

    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    save_path = "images/filter_" + str(k) + ".png"
    savefig(plt, save_path)