"""
It is a modified version of the official implementation of
'Scale-Equivariant Steerable Networks'
Paper: https://arxiv.org/abs/1910.11093
Code: https://github.com/ISosnovik/sesn

------------------------------------------------------------
MIT License

Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .ses_basis import steerable_A, normalize_basis_by_min_scale


def cast_to_cpu_cuda_tensor(input, reference_tensor):
    if reference_tensor.is_cuda and not input.is_cuda:
        input = input.cuda()
    if not reference_tensor.is_cuda and input.is_cuda:
        input = input.cpu()
    return input

def get_sqroot_patch_energy(input, kernel_size, padding, stride= 1, norm_p = 2.0):
    num_channels = input.shape[1]
    input_squared  = torch.pow(input, norm_p)

    # Taken from https://discuss.pytorch.org/t/manual-2d-convolution-per-channel/83907/2
    pool_filters = cast_to_cpu_cuda_tensor(torch.ones((kernel_size, kernel_size)).float().unsqueeze(0).unsqueeze(0), input)
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
        norm_kernel_wt    = F.normalize(kernel_wt.reshape((-1, K*K)), dim= 1).reshape((out_channels, in_channels_grp, K, K)) # Cout x Cin x H x W
    else:
        norm_kernel_wt    = kernel_wt

    # First get the sq_root_patch_energy
    if normalize_patches:
        sqroot_patch_energy   = get_sqroot_patch_energy(input, kernel_size= K, padding= padding, stride= stride, norm_p = norm_p) # B x Cin x Hout x Wout

    output       = cast_to_cpu_cuda_tensor(torch.zeros((batch, out_channels, Hout, Wout)), reference_tensor= input)
    index_input  = torch.arange(groups+1)*(in_channels//groups)
    index_output = torch.arange(groups+1)*(out_channels//groups)

    for i in range(groups):
        inp_group             = input[:, index_input[i]:index_input[i+1]]

        # Taken from
        # https://discuss.pytorch.org/t/visualizing-the-outputs-of-kernels-instead-of-the-outputs-of-filters/137266/4
        norm_kernel_wt_interm = norm_kernel_wt[index_output[i]:index_output[i+1]].reshape(out_channels_grp * in_channels_grp, 1, K, K)
        output_grp            = F.conv2d(inp_group, norm_kernel_wt_interm, padding= padding, stride= stride, groups= in_channels_grp) # B x Cout*Cin x Hout x Wout
        output_grp            = output_grp.view(batch, in_channels_grp, out_channels_grp, Hout, Wout).permute(0, 2, 1, 3, 4).reshape(batch, out_channels_grp, in_channels_grp, Hout, Wout)

        """
        # Takes more memory because of torch.unfold
        inp_unf               = torch.nn.functional.unfold(inp_group, (K, K), padding= padding, stride= stride)  # B x Cin*K*K X Hout*Wout
        inp_unf_interm        = inp_unf.transpose(1, 2).unsqueeze(1)  # B x 1 x Hout*Wout x Cin*K*K

        norm_kernel_wt_interm = norm_kernel_wt[index_output[i]:index_output[i+1]].view(out_channels_grp, -1).unsqueeze(0).unsqueeze(2) # 1 x Cout_grp x 1 x Cin*K*K
        out_unf    = torch.mul(inp_unf_interm, norm_kernel_wt_interm).reshape(batch, out_channels_grp, inp_unf_interm.shape[2], in_channels_grp, -1) # B x Cout_grp x Hout*Wout x Cin x K*K
        output_grp = torch.sum(out_unf, dim= 4).transpose(2, 3).reshape(batch, out_channels_grp, in_channels_grp, Hout, Wout) # B x Cout_grp x Cin_grp x Hout x Wout
        if debug:
            print("\nout_unf.shape before sum", out_unf.shape)
        """

        if debug:
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


class SESConv_Z2_H(nn.Module):
    '''Scale Equivariant Steerable Convolution: Z2 -> (S x Z2)
    [B, C, H, W] -> [B, C', S, H', W']

    Args:
        in_channels: Number of channels in the x_in image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the x_in
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, kernel_size, effective_size,
                 scales=[1.0], stride=1, padding=0, bias=False, padding_mode='constant', dilation= 1, norm_per_scale= True, rescale_basis= False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = [round(s, 3) for s in scales]
        self.num_scales = len(scales)
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.norm_per_scale = norm_per_scale
        self.rescale_basis  = rescale_basis

        basis = steerable_A(kernel_size, scales, effective_size, **kwargs)
        # basis = normalize_basis_by_min_scale(basis, norm_per_scale= self.norm_per_scale, rescale_basis= rescale_basis)
        self.register_buffer('basis', basis)

        self.num_funcs = self.basis.size(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        basis = self.basis.view(self.num_funcs, -1)
        kernel = self.weight @ basis
        kernel = kernel.view(self.out_channels, self.in_channels,
                             self.num_scales, self.kernel_size, self.kernel_size)
        kernel = kernel.permute(0, 2, 1, 3, 4).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        # convolution
        if self.padding > 0:
            x = F.pad(x, 4 * [self.padding], mode= self.padding_mode)
        y = F.conv2d(x, kernel, bias= None, stride= self.stride, dilation= self.dilation)
        B, C, H, W = y.shape
        y = y.view(B, self.out_channels, self.num_scales, H, W)

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)

        return y

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, scales={scales}, size={kernel_size}, eff_size={effective_size}, stride={stride}, padding={padding}'
        return s.format(**self.__dict__)


class SESConv_H_H(nn.Module):
    '''Scale Equivariant Steerable Convolution: (S x Z2) -> (S x Z2)
    [B, C, S, H, W] -> [B, C', S', H', W']

    Args:
        in_channels: Number of channels in the x_in image
        out_channels: Number of channels produced by the convolution
        scale_size: Size of scale filter
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the x_in
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, scale_size, kernel_size, effective_size,
                 scales=[1.0], stride=1, padding=0, bias=False, padding_mode='constant', norm_per_scale= True, rescale_basis= False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_size = scale_size
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = [round(s, 3) for s in scales]
        self.num_scales = len(scales)
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.norm_per_scale = norm_per_scale
        self.rescale_basis  = rescale_basis

        basis = steerable_A(kernel_size, scales, effective_size, **kwargs)
        # basis = normalize_basis_by_min_scale(basis, norm_per_scale= self.norm_per_scale, rescale_basis= rescale_basis)
        self.register_buffer('basis', basis)

        self.num_funcs = self.basis.size(0)

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, scale_size, self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # get kernel
        basis = self.basis.view(self.num_funcs, -1)
        kernel = self.weight @ basis
        kernel = kernel.view(self.out_channels, self.in_channels, self.scale_size,
                             self.num_scales, self.kernel_size, self.kernel_size)

        # expand kernel
        kernel = kernel.permute(3, 0, 1, 2, 4, 5).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.scale_size,
                             self.kernel_size, self.kernel_size)

        # calculate padding
        if self.scale_size != 1:
            value = x.mean()
            x = F.pad(x, [0, 0, 0, 0, 0, self.scale_size - 1])

        output = 0.0
        for i in range(self.scale_size):
            x_ = x[:, :, i:i + self.num_scales]
            # expand X
            B, C, S, H, W = x_.shape
            x_ = x_.permute(0, 2, 1, 3, 4).contiguous()
            x_ = x_.view(B, -1, H, W)
            if self.padding > 0:
                x_ = F.pad(x_, 4 * [self.padding], mode=self.padding_mode)
            output += F.conv2d(x_, kernel[:, :, i], groups=S, stride=self.stride)

        # squeeze output
        B, C_, H_, W_ = output.shape
        output = output.view(B, S, -1, H_, W_)
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, scales={scales}, size={kernel_size}, eff_size={effective_size}, stride={stride}, padding={padding}'
        return s.format(**self.__dict__)


class SESConv_H_H_1x1(nn.Module):

    def __init__(self, in_channels, out_channels, scale_size=1, stride=1, num_scales=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_size = scale_size
        self.stride = (1, stride, stride)
        if scale_size > 1:
            # workaround for being compatible with the old-style weight init
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, scale_size, 1, 1))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        # nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        weight = self.weight
        if len(weight.shape) == 4:
            weight = weight[:, :, None]
        pad = self.scale_size - 1
        return F.conv3d(x, weight, padding=[pad, 0, 0], stride=self.stride)[:, :, pad:]

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, scale_size={scale_size}, kernel_size=(1, 1),  stride={stride}'
        return s.format(**self.__dict__)

class SESMax_Scale(nn.Module):
    """
    Scale Equivariant Steerable Max over Scale
    [B, C, S, H, W] -> [B, C, H, W]
    Calculates max over scale
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_in):
        if isinstance(x_in, list):
            x = x_in[0]
        else:
            x = x_in

        output = x.max(2)[0]

        if isinstance(x_in, list):
            x_in[0] = output
            return x_in
        else:
            return output

class SESCopy_Scale(nn.Module):
    """
    Scale Equivariant Steerable Copy for different scale
    [B, C, H, W] -> [B, C, S, H, W]
    Copies x_in to be applied over different scales
    """
    def __init__(self, scales= [1.0]):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)

    def forward(self, x_in):
        if isinstance(x_in, list):
            x = x_in[0]
        else:
            x = x_in

        output = x.unsqueeze(2).repeat(1, 1, self.num_scales, 1, 1)

        if isinstance(x_in, list):
            x_in[0] = output
            return x_in
        else:
            return output

    def extra_repr(self):
        s = 'num_scales= {num_scales}'
        return s.format(**self.__dict__)

def ses_max_projection(x):
    return x.max(2)[0]
