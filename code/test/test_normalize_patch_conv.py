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


def get_sqroot_patch_energy(input, kernel_size, padding, stride= 1, norm_p = 2.0):
    num_channels = input.shape[1]
    input_squared  = torch.pow(input, norm_p)

    # Taken from https://discuss.pytorch.org/t/manual-2d-convolution-per-channel/83907/2
    pool_filters = torch.ones((kernel_size, kernel_size)).float().unsqueeze(0).unsqueeze(0)
    pool_filters = pool_filters.repeat((num_channels, 1, 1, 1))
    patch_energy = F.conv2d(input_squared, pool_filters.detach(), padding= padding, stride= stride, groups= num_channels)

    if norm_p == 1.0:
        return patch_energy
    elif norm_p == 2.0:
        return torch.sqrt(patch_energy)

def convolution_with_normalized_patches(input, kernel_wt, padding= 0, stride= 1, groups= 1, normalize_filters= True, normalize_patches= True, norm_p= 2.0, eps= 1e-3, debug= False):
    if debug:
        print("\nInside function ...")

    # kernel_wt with shape Cout x Cin x K x K
    out_channels, _, K, K = kernel_wt.shape
    batch, in_channels, H, W                  = input.shape
    Hout = (H + 2*padding - 2*(K//2))//stride
    Wout = (W + 2*padding - 2*(K//2))//stride
    out_channels_grp = out_channels//groups
    in_channels_grp  = in_channels//groups

    # Normalize kernel_wt
    if normalize_filters:
        norm_kernel_wt    = F.normalize(kernel_wt.reshape((-1, K*K)), dim= 1).reshape((out_channels, in_channels, K, K)) # Cout x Cin x H x W
    else:
        norm_kernel_wt    = kernel_wt

    # First get the sq_root_patch_energy
    if normalize_patches:
        sqroot_patch_energy   = get_sqroot_patch_energy(input, kernel_size= K, padding= padding, stride= stride, norm_p = norm_p) # B x Cin x Hout x Wout

    output       = torch.zeros((batch, out_channels, Hout, Wout))
    index_input  = torch.arange(groups+1)*(in_channels//groups)
    index_output = torch.arange(groups+1)*(out_channels//groups)

    for i in range(groups):
        inp_group             = input[:, index_input[i]:index_input[i+1]]
        inp_unf               = torch.nn.functional.unfold(inp_group, (K, K), padding= padding, stride= stride)  # B x Cin*K*K X Hout*Wout
        inp_unf_interm        = inp_unf.transpose(1, 2).unsqueeze(1)  # B x 1 x Hout*Wout x Cin*K*K

        norm_kernel_wt_interm = norm_kernel_wt[index_output[i]:index_output[i+1]].view(out_channels_grp, -1).unsqueeze(0).unsqueeze(2) # 1 x Cout_grp x 1 x Cin*K*K
        out_unf    = torch.mul(inp_unf_interm, norm_kernel_wt_interm).reshape(batch, out_channels_grp, inp_unf_interm.shape[2], in_channels_grp, -1) # B x Cout_grp x Hout*Wout x Cin x K*K
        output_grp = torch.sum(out_unf, dim= 4).transpose(2, 3).reshape(batch, out_channels_grp, in_channels_grp, Hout, Wout) # B x Cout_grp x Cin_grp x Hout x Wout
        if debug:
            print("\nout_unf.shape before sum", out_unf.shape)
            print("output_grp.shape before sum", output_grp.shape)

        # Divide by sqroot_patch_energy
        if normalize_patches:
            if debug:
                print("Normalizing by patch energy")
                print("patch energy.shape", sqroot_patch_energy.shape)
            output_div_by_energy  = output_grp / (sqroot_patch_energy[:, index_input[i]:index_input[i+1]].unsqueeze(1) + eps)
        else:
            output_div_by_energy  = output_grp

        output_div_by_energy = torch.sum(output_div_by_energy, dim= 2) # B x Cout x Hout x Wout
        if debug:
            print("output_div_by_energy shape", output_div_by_energy.shape)

        output[:, index_output[i]:index_output[i+1]] = output_div_by_energy

    return output

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

batch        = 100
in_channels  = 3
groups       = 3
out_channels = 3*groups

K = 3
padding = K//2
H       = 28
W       = 36
stride  = 2
normalize_patches = False
debug   = True

Hout = (H + 2*padding - 2*(K//2))//stride
Wout = (W + 2*padding - 2*(K//2))//stride

inp = torch.randn(batch, in_channels, H, W)
# inp = F.pad(inp, 2 * [1, 1])
print(inp.shape)
w   = torch.randn(out_channels, in_channels//groups, K, K)

out2 = convolution_with_normalized_patches(inp, kernel_wt= w, padding= padding, stride= stride, groups= groups, normalize_filters= False, normalize_patches= normalize_patches, debug= debug)
print(out2.shape)
print((torch.nn.functional.conv2d(inp, w, padding= padding, stride= stride, groups= groups) - out2).abs().max())