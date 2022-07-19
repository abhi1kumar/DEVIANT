"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.set_printoptions   (precision= 4, suppress= True)
torch.set_printoptions(precision= 4, sci_mode= False)

from lib.projective.ses_conv import *

def test_one_case(batch, in_channels, out_channels, H, W, stride, groups, normalize_filters, normalize_patches):
    print("")
    if groups > 1:
        print("Carrying out the group stuff")

    kernel_size = K
    effective_size = K-2
    padding = K//2
    Hout = (H + 2*padding - 2*(K//2))//stride
    Wout = (W + 2*padding - 2*(K//2))//stride

    inp = torch.randn(batch, in_channels, H, W)
    #inp = torch.arange(batch*in_channels*H*W).float().reshape(batch, in_channels, H, W)
    w   = torch.ones(out_channels, in_channels//groups, K, K)


    ideal_output = torch.nn.functional.conv2d(inp, w, padding= padding, stride= stride, groups= groups)
    print("Ideal output shape=", ideal_output.shape)

    out2 = convolution_with_normalized_patches(inp, kernel_wt= w, padding= padding, stride= stride, groups= groups, normalize_filters= normalize_filters, normalize_patches= normalize_patches, debug= debug)
    print("Norm Patches shape=", out2.shape)
    print("Max Difference Ten=", (ideal_output[:, :2] - out2[:, :2]).abs().max())

    model_ses = nn.Sequential(SESCopy_Scale(scales= sesn_scales), SESConv_H_H(in_channels= in_channels, out_channels= out_channels, kernel_size= kernel_size, effective_size=effective_size,
                                                       scales= sesn_scales, stride= stride, padding= padding, bias= False, padding_mode= 'constant', norm_per_scale= False, dilation= 1,
                                                       scale_size= 1, rescale_basis= False))
    print("Model output shape=", model_ses(inp).shape)

seed     = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

sesn_scales  = [0.83, 1.0, 1.2]

batch        = 10
groups       = 2
in_channels  = 2
out_channels = 2
K       = 5
H       = 6
W       = 8
stride  = 1
normalize_patches = False
normalize_filters = False
debug = False



test_one_case(batch= batch, in_channels= in_channels, out_channels= out_channels, H= H, W= W, groups= 1, stride= 1, normalize_filters= normalize_filters, normalize_patches= normalize_patches)
test_one_case(batch= batch, in_channels= in_channels, out_channels= out_channels, H= H, W= W, groups= 1, stride= 2, normalize_filters= normalize_filters, normalize_patches= normalize_patches)
test_one_case(batch= batch, in_channels= in_channels, out_channels= out_channels, H= H, W= W, groups= groups, stride= 2, normalize_filters= normalize_filters, normalize_patches= normalize_patches)
test_one_case(batch= batch, in_channels= in_channels, out_channels= out_channels, H= H, W= W, groups= groups, stride= 1, normalize_filters= normalize_filters, normalize_patches= normalize_patches)



"""
N, C, H, W = torch.randint(1, 100, (4,))

x = torch.rand(N, C, H, W)

out_channels = 8
kw = 3
conv = nn.Conv2d(C, out_channels, kw, 1, 1, bias=False)
out = conv(x)
#print(out.shape)
# > torch.Size([1, 8, 24, 24])

conv_grouped = nn.Conv2d(C, C*out_channels, kw, 1, 1, groups=C, bias=False)
with torch.no_grad():
    conv_grouped.weight.copy_(conv.weight.permute(1, 0, 2, 3).reshape(C*out_channels, 1, kw, kw))
out_grouped = conv_grouped(x)
#print(out_grouped.shape)
# > torch.Size([1, 32, 24, 24])

out_grouped = out_grouped.view(N, C, out_channels, H, W).permute(0, 2, 1, 3, 4).reshape(N, C*out_channels, H, W)

# manually reduce
idx = torch.arange(out_channels)
idx = torch.repeat_interleave(idx, C)
idx = idx[None, :, None, None].expand(N, -1, H, W)
out_grouped_reduced = torch.zeros_like(out)
out_grouped_reduced.scatter_add_(dim=1,index=idx, src=out_grouped)

# check error
# print(torch.allclose(out_grouped_reduced, out, atol=5e-6), (out_grouped_reduced - out).abs().max())
"""


