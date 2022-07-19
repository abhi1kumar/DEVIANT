"""
    Projective Convolutions for the network
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import copy

PI = np.pi
EPS= 0.001

def cast_to_cpu_cuda_tensor(input, reference_tensor):
    if reference_tensor.is_cuda and not input.is_cuda:
        input = input.cuda()
    if not reference_tensor.is_cuda and input.is_cuda:
        input = input.cpu()
    return input

def convert_HWC_to_CHW(input):
    """
        Converts tensor or numpy array input from HWC to CHW format
    """
    if type(input) == np.ndarray:
        ndim = input.ndim
        if ndim == 2:
            output = input[np.newaxis, :, :]
        elif ndim == 3:
            # HWC --> CHW
            output = input.transpose(2, 0, 1)
        elif ndim == 4:
            # BHWC --> BCHW
            output = input.transpose(0, 3, 1, 2)

    return output

def convert_CHW_to_HWC(input):
    """
        Converts tensor or numpy array input from HWC to CHW format
    """
    if type(input) == np.ndarray:
        ndim = input.ndim
        if ndim == 3:
            # CHW --> HWC
            output = input.transpose(1, 2, 0)
        elif ndim == 4:
            # BCHW --> BHWC
            output = input.transpose(0, 2, 3, 1)

    return output


def cartesian_to_polar(X, Y):
    # X = torch.Tensor([h, w])
    # Y = torch.Tensor(h, w])
    # From https://discuss.pytorch.org/t/polar-coordinates-transformation-layer/66717/2
    R = (X ** 2 + Y ** 2).sqrt()
    # With consideration of angle.
    Theta = torch.atan2(Y, (X + EPS))

    # Theta is in range [-\pi, pi]. Bring in range [0, 2 \pi]
    Theta[Theta < 0] += 2.0 * PI

    return R, Theta

def polar_to_cartesian(R, Theta):
    X = R * torch.cos(Theta)
    Y = R * torch.sin(Theta)

    cartesian = torch.stack([X, Y], dim= -1)

    return X, Y

def geometric_center(h, w):
    geom_x = (w-1)/2.0
    geom_y = (h-1)/2.0

    return geom_x, geom_y

def get_centers(center, input_feature_map):

    if type(input_feature_map) == torch.Tensor:
        batch, c, h, w = input_feature_map.shape
        geom_x, geom_y = geometric_center(h= h, w= w)

        if center is None:
            center_x = geom_x * torch.ones((batch, ))
            center_y = geom_y * torch.ones((batch, ))
        else:
            center_x = center[:, 0]
            center_y = center[:, 1]

        center_x     = center_x.type(input_feature_map.dtype)
        center_y     = center_y.type(input_feature_map.dtype)
        center_x     = cast_to_cpu_cuda_tensor(center_x, input_feature_map)
        center_y     = cast_to_cpu_cuda_tensor(center_y, input_feature_map)

    elif type(input_feature_map) == np.ndarray:
        c, h, w = input_feature_map.shape
        geom_x, geom_y = geometric_center(h= h, w= w)

        if center is None:
            center_x = geom_x
            center_y = geom_y
        else:
            center_x = center[0]
            center_y = center[1]

    return center_x, center_y

def get_max_coordinate(center_dim, max_of_this_dim):
    if type(center_dim) == torch.Tensor:
        max_coordinate = torch.max(torch.cat([center_dim.unsqueeze(0), max_of_this_dim - center_dim.unsqueeze(0)]))

    return max_coordinate

def logical_or(input1, input2):

    return input1.byte() |  input2.byte()

def logical_and(input1, input2):

    return input1.byte() &  input2.byte()

def logical_not(input1):

    return ~input1.byte()

def index_4D_tensor_by_two_coordinates(input, index_x, index_y):
    """
    Indexes a 4D input tensor by two coordinates. We reshape the common dimensions and then use gather along second dimension.
     -----------> X
     |
     |
     V
     Y
    :param input:    input tensor of shape  batch x channel x   h   x w
    :param index_x:  long tensor of shape   batch x channel x out_h x out_w   or    batch x out_h x out_w   or   out_h x out_w
    :param index_y:  long tensor of shape   batch x channel x out_h x out_w   or    batch x out_h x out_w   or   out_h x out_w
    :return:
           output:   output tensor of shape batch x channel x out_h x out_w
    """
    batch, channel, h, w = input.shape

    if index_x.dim() == 2:
        # Add batch and channels
        index_x = index_x.unsqueeze(0).unsqueeze(0).repeat(batch, channel, 1, 1)
        index_y = index_y.unsqueeze(0).unsqueeze(0).repeat(batch, channel, 1, 1)

    elif index_x.dim() == 3:
        # Add channels
        index_x = index_x.unsqueeze(1).repeat(1, channel, 1, 1)
        index_y = index_y.unsqueeze(1).repeat(1, channel, 1, 1)

    if index_x.dim() == 4:
        _, channel, grid_h, grid_w = index_x.shape

    # Reshape input in 2D
    input_2d = input  .reshape(batch*channel, h*w)  # batch*channel x h*w

    # Get the flattened 2D index
    index_x  = index_x.reshape(batch*channel, -1)   # batch*channel x grid_h*grid_w
    index_y  = index_y.reshape(batch*channel, -1)   # batch*channel x grid_h*grid_w

    # Get illegal indices outside the range
    invalid_mask_for_index_x = logical_or(index_x < 0, index_x >= w)                          # batch*channel x grid_h*grid_w
    invalid_mask_for_index_y = logical_or(index_y < 0, index_y >= h)                          # batch*channel x grid_h*grid_w
    invalid_mask             = logical_or(invalid_mask_for_index_x, invalid_mask_for_index_y) # batch*channel x grid_h*grid_w
    valid_mask_2d            = logical_not(invalid_mask)

    index_2d = index_y * w + index_x                # batch*channel x grid_h*grid_w

    # Multiply with mask so that gather does not throw an error.
    index_2d = index_2d* valid_mask_2d.long()

    # Reference:
    # https://pytorch.org/docs/master/generated/torch.gather.html#torch.gather
    # For 3D array: out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    # For 2D array: out[i][j]    = input[i][index[i][j]]
    output_2d = torch.gather(input_2d, dim= 1, index= index_2d)  # batch*channel x grid_h*grid_w

    # Make sure the output corresponding to invalid indices are suppressed to zero
    output_2d = output_2d * valid_mask_2d.type(output_2d.dtype)

    output    = output_2d.reshape(batch, channel, grid_h, grid_w)

    return output

def interpolate_feature_map(input_feature_map, grid_normalized, mode= "bilinear"):
    batch, channel, h, w = input_feature_map.shape
    _, grid_h, grid_w, _ = grid_normalized.shape

    output               = torch.zeros((batch, channel, grid_h, grid_w))
    output               = cast_to_cpu_cuda_tensor(output, input_feature_map)

    grid_normalized      = cast_to_cpu_cuda_tensor(grid_normalized, input_feature_map)
    grid_normalized_x    = grid_normalized[:, :, :, 0]
    grid_normalized_y    = grid_normalized[:, :, :, 1]

    # Unnormalize
    grid_x = (grid_normalized_x + 1.0) * w/2.0
    grid_y = (grid_normalized_y + 1.0) * h/2.0

    grid_x = grid_x.unsqueeze(1).repeat(1, channel, 1, 1)   # batch x channel x grid_h x grid_w
    grid_y = grid_y.unsqueeze(1).repeat(1, channel, 1, 1)   # batch x channel x grid_h x grid_w

    left   = torch.floor(grid_x).long()               # batch x channel x grid_h x grid_w
    top    = torch.floor(grid_y).long()               # batch x channel x grid_h x grid_w
    right  = left + 1                                 # batch x channel x grid_h x grid_w
    bottom = top  + 1                                 # batch x channel x grid_h x grid_w

    top_left_pixel     = index_4D_tensor_by_two_coordinates(input= input_feature_map, index_x= left , index_y= top)    # batch x channel x grid_h x grid_w
    top_right_pixel    = index_4D_tensor_by_two_coordinates(input= input_feature_map, index_x= right, index_y= top)    # batch x channel x grid_h x grid_w
    bottom_left_pixel  = index_4D_tensor_by_two_coordinates(input= input_feature_map, index_x= left , index_y= bottom) # batch x channel x grid_h x grid_w
    bottom_right_pixel = index_4D_tensor_by_two_coordinates(input= input_feature_map, index_x= right, index_y= bottom) # batch x channel x grid_h x grid_w

    dist_left          = grid_x                               - left.type(input_feature_map.dtype)          # batch x channel x grid_h x grid_w
    dist_right         = right.type(input_feature_map.dtype)  - grid_x                                      # batch x channel x grid_h x grid_w
    dist_top           = grid_y                               - top.type(input_feature_map.dtype)           # batch x channel x grid_h x grid_w
    dist_bottom        = bottom.type(input_feature_map.dtype) - grid_y                                      # batch x channel x grid_h x grid_w

    if mode == "bilinear":
        # See https://en.wikipedia.org/wiki/Bilinear_interpolation
        # The product of the value at the desired point (black) and the entire area is equal to the
        # sum of the products of the value at each corner and the partial area diagonally opposite
        # the corner (corresponding colours)
        output = dist_bottom * dist_right * top_left_pixel + \
                 dist_bottom * dist_left  * top_right_pixel + \
                 dist_top    * dist_right * bottom_left_pixel + \
                 dist_top    * dist_left  * bottom_right_pixel

    elif mode == "nearest":
        # TODO
        pass

    return output

def cartesian_to_log_polar_feature_map(input_feature_map, center_cartesian_feature_map= None, padding_mode="zeros", interpolation_method="deterministic", upsample_factor_for_log_polar_output= 1.0):
    """
    Converts cartesian feature map to log polar
    :param input_feature_map: Tensor array [b x c x h x w] or
                              numpy  array [    c x h x w]
    :return:
        log_polar_feature_map: Tensor array [b x c x h x w] or
                               numpy  array [    c x h x w]
    """

    if type(input_feature_map) == torch.Tensor:

        # Get the center first
        center_x, center_y = get_centers(center_cartesian_feature_map, input_feature_map)

        batch, c, h, w = input_feature_map.shape
        out_h          = int(h * upsample_factor_for_log_polar_output)
        out_w          = int(w * upsample_factor_for_log_polar_output)

        x_max        = get_max_coordinate(center_dim= center_x, max_of_this_dim= w)
        y_max        = get_max_coordinate(center_dim= center_y, max_of_this_dim= h)
        r_max        = torch.sqrt(x_max*x_max + y_max*y_max)

        log_r_max    = torch.log(r_max + EPS)
        theta_max    = 2.0 * PI
        log_r_upsamp = out_w/log_r_max
        theta_upsamp = out_h/theta_max

        # Make log_r, theta vector first
        log_r_vector = torch.arange(out_w).type(input_feature_map.dtype)/ log_r_upsamp
        theta_vector = torch.arange(out_h).type(input_feature_map.dtype)/ theta_upsamp

        # Make log_r, theta grid now
        log_r_grid   = log_r_vector.unsqueeze(0).repeat(theta_vector.shape[0], 1)
        theta_grid   = theta_vector.unsqueeze(1).repeat(1, log_r_vector.shape[0])

        # This is centered at supplied center
        r_grid            = torch.exp(log_r_grid)
        x_grid, y_grid    = polar_to_cartesian(R= r_grid, Theta= theta_grid)    # out_h x out_w

        # Centered at top-left
        x_grid_top_center = x_grid.unsqueeze(0).repeat(batch, 1, 1) + center_x.unsqueeze(1).unsqueeze(1)  # batch x out_h x out_w
        y_grid_top_center = y_grid.unsqueeze(0).repeat(batch, 1, 1) + center_y.unsqueeze(1).unsqueeze(1)  # batch x out_h x out_w

        # Bring x_grid_normalized and y_grid_normalized in the range [-1, 1]
        x_grid_normalized = x_grid_top_center * 2.0/w - 1.0   # batch x out_h x out_w
        y_grid_normalized = y_grid_top_center * 2.0/h - 1.0   # batch x out_h x out_w

        # grid = [grid_x, grid_y] convention
        #  -----------> X
        #  |
        #  |
        #  V
        #  Y
        grid_final = torch.cat((x_grid_normalized.unsqueeze(3), y_grid_normalized.unsqueeze(3)), dim= 3)    # batch x out_h x out_w x 2
        grid_final = grid_final.detach()

        # Log Polar coordinate sample
        if interpolation_method == "deterministic":
            log_polar_feature_map = interpolate_feature_map(input_feature_map, grid_normalized= grid_final)
        else:
            log_polar_feature_map = F.grid_sample(input_feature_map, grid_final, mode= "bilinear", padding_mode= padding_mode)

    elif type(input_feature_map) == np.ndarray:

        # Get the center first
        center_x, center_y = get_centers(center_cartesian_feature_map, input_feature_map)

        # CHW --> HWC
        input_feature_map = convert_CHW_to_HWC(input_feature_map)

        log_polar_feature_map = cv2.logPolar(input_feature_map    , center= (center_x, center_y), M= min(center_x, center_y), flags= 0)
        log_polar_feature_map = convert_HWC_to_CHW(log_polar_feature_map)

    return log_polar_feature_map

def log_polar_to_cartesian_feature_map(log_polar_feature_map, center_cartesian_feature_map= None, padding_mode="zeros", interpolation_method="deterministic", upsample_factor_for_log_polar_output= 1.0):
    """
    Converts cartesian feature map to log polar
    :param log_polar_feature_map: Tensor array [b x c x h x w] or
                              numpy  array [    c x h x w]
    :return:
        cartesian_feature_map: Tensor array [b x c x h x w] or
                               numpy  array [    c x h x w]
    """

    if type(log_polar_feature_map) == torch.Tensor:

        batch, c, h, w = log_polar_feature_map.shape
        h_out          = int(h / upsample_factor_for_log_polar_output)
        w_out          = int(w / upsample_factor_for_log_polar_output)

        # Get the center first in output resolution
        temp_feat_map  = torch.zeros((batch, 1, h_out, w_out)).type(log_polar_feature_map.dtype)
        center_x, center_y = get_centers(center_cartesian_feature_map, input_feature_map= temp_feat_map)


        x_max_out    = get_max_coordinate(center_dim= center_x, max_of_this_dim= w_out).float()
        y_max_out    = get_max_coordinate(center_dim= center_y, max_of_this_dim= h_out).float()
        r_max_out    = torch.sqrt(x_max_out*x_max_out + y_max_out*y_max_out)

        log_r_max    = torch.log(r_max_out* upsample_factor_for_log_polar_output + EPS)
        theta_max    = 2.0 * PI
        log_r_upsamp = w/log_r_max
        theta_upsamp = h/theta_max

        # Make x, y vector first
        x_vector_out = torch.arange(w_out).type(log_polar_feature_map.dtype)
        y_vector_out = torch.arange(h_out).type(log_polar_feature_map.dtype)

        # Make x, y grid now
        x_grid_out   = x_vector_out.unsqueeze(0).unsqueeze(0).repeat(batch, y_vector_out.shape[0], 1) - center_x.unsqueeze(1).unsqueeze(1)  # batch x h_out x w_out  each entry in [0, h_out]
        y_grid_out   = y_vector_out.unsqueeze(1).unsqueeze(0).repeat(batch, 1, x_vector_out.shape[0]) - center_y.unsqueeze(1).unsqueeze(1)  # batch x h_out x w_out  each entry in [0, h_out]

        # Upsample each [0, h_out] entry --> [0, h] range
        x_grid   = x_grid_out * upsample_factor_for_log_polar_output # batch x h_out x w_out
        y_grid   = y_grid_out * upsample_factor_for_log_polar_output # batch x h_out x w_out

        r_grid, theta_grid = cartesian_to_polar(X= x_grid, Y= y_grid)
        log_r_grid         = torch.log(r_grid + EPS)

        # Centered at top-left
        # log_r is in range [0, log_r_max]. Bring it in the range [0, w]
        # theta_grid is in range [0, 2\pi]. Bring it in the range [0, h]
        log_r_grid_top_center = log_r_grid * log_r_upsamp               # batch x h_out x w_out
        theta_grid_top_center = theta_grid * theta_upsamp               # batch x h_out x w_out

        # Bring log_r_grid and theta_grid in the range [-1, 1]
        log_r_grid_normalized = log_r_grid_top_center * 2.0/w - 1.0     # batch x h_out x w_out
        theta_grid_normalized = theta_grid_top_center * 2.0/h - 1.0     # batch x h_out x w_out

        # grid = [grid_x, grid_y] convention
        # grid = [log_r_grid, theta_grid] convention
        #  -----------> X = log_r
        #  |
        #  |
        #  V
        #  Y = theta
        grid_final = torch.cat((log_r_grid_normalized.unsqueeze(3), theta_grid_normalized.unsqueeze(3)), dim= 3)    # batch x h_out x w_out x 2
        grid_final = grid_final.detach()

        # Inverse Log polar sample
        if interpolation_method == "deterministic":
            cartesian_feature_map = interpolate_feature_map(log_polar_feature_map, grid_normalized= grid_final)
        else:
            cartesian_feature_map = F.grid_sample(log_polar_feature_map, grid_final, mode= "bilinear", padding_mode= padding_mode)

    elif type(log_polar_feature_map) == np.ndarray:

        # Get the center first
        center_x, center_y = get_centers(center_cartesian_feature_map, input_feature_map= log_polar_feature_map)

        # CHW --> HWC
        log_polar_feature_map = convert_CHW_to_HWC(log_polar_feature_map)

        cartesian_feature_map = cv2.logPolar(log_polar_feature_map    , center= (center_x, center_y), M= min(center_x, center_y), flags= cv2.WARP_INVERSE_MAP)
        cartesian_feature_map = convert_HWC_to_CHW(cartesian_feature_map)

    return cartesian_feature_map

def our_interpolate(input, scale_factor, mode= "bilinear"):
    if scale_factor == 1.0:
        return input
    else:
        return F.interpolate(input, scale_factor= scale_factor, mode= mode)

class DilatedConvolution(torch.nn.Module):

    def __init__(self, init_conv, scales= [1.0]):
        super(DilatedConvolution, self).__init__()
        # Usual conv
        self.in_channels    = init_conv.in_channels
        self.out_channels   = init_conv.out_channels
        self.kernel_size    = init_conv.kernel_size
        self.num_scales     = len(scales)

        self.stride         = init_conv.stride[0]
        self.padding        = init_conv.padding[0]
        # We use the same weight to obtain dilated outputs
        self.weight         = torch.nn.Parameter(init_conv.weight)

    def forward(self, input):
        for i in range(self.num_scales):
            dilation   = i+1
            padding    = self.padding * (i+1)
            dil_output = F.conv2d(input, weight = self.weight, bias=None, stride= self.stride, padding= padding, dilation= dilation)

            if i == 0:
                max_output = dil_output
            else:
                max_output = torch.max(torch.stack((max_output, dil_output)), dim= 0)[0]

        return max_output

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, num_scales={num_scales}, size={kernel_size}, stride={stride}, padding={padding}'
        return s.format(**self.__dict__)

class LogPolarConvolution(torch.nn.Module):

    def __init__(self, init_conv, do_upsample= True, upsample_factor= 4.0):
        super(LogPolarConvolution, self).__init__()
        self.do_upsample     = do_upsample
        self.upsample_factor = upsample_factor

        # Usual conv
        self.conv            = copy.deepcopy(init_conv)

    def forward(self, input, center_input= None):
        if self.do_upsample and self.upsample_factor != 1:
            # upsample input feature maps
            input_upsamp       = our_interpolate(input, scale_factor= self.upsample_factor)
            if center_input is not None:
                center_upsamp  = center_input * self.upsample_factor
            else:
                center_upsamp  = None
        else:
            input_upsamp       = input
            center_upsamp      = center_input

        log_polar_input_upsamp = cartesian_to_log_polar_feature_map(input_upsamp, center_cartesian_feature_map= center_upsamp, interpolation_method= "pytorch")
        log_polar_output       = self.conv(log_polar_input_upsamp)
        output_upsamp          = log_polar_to_cartesian_feature_map(log_polar_output, center_cartesian_feature_map= center_upsamp, interpolation_method= "pytorch")

        if self.do_upsample and self.upsample_factor != 1:
            # feature maps were upsampled, downsample the output feature map
            output   = our_interpolate(output_upsamp, scale_factor= 1.0/self.upsample_factor)
        else:
            output   = output_upsamp

        return output