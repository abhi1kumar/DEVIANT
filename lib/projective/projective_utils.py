
import os, sys
sys.path.append(os.getcwd())

import logging
import copy
from collections import OrderedDict

from torchvision.models.resnet import BasicBlock
from torchvision.models.densenet import _DenseLayer
from lib.backbones.dla import Tree

from lib.projective.ses_conv import *
from lib.projective.log_polar_conv import LogPolarConvolution, DilatedConvolution
from lib.projective.sesn_utils import calculate_new_weights

def replace_single_layer(network, child, child_name, scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= "replicate"):
    """
    Replace single layer by SES (Scale Equivariant Steerable) counterpart
    :param network:    network object to replace
    :param child:      network child object
    :param child_name: name
    :param scales:     sesn scales
    :param first_conv: handles first conv specially
    :return:
    """
    new_layer = None
    # sesn_padding_mode   = "constant" # "circular"

    if first_conv:
        # Replace the first conv by SESConv_Z2_H
        in_channels    = child.in_channels
        out_channels   = child.out_channels
        effective_size = child.kernel_size[0]
        kernel_size    = child.kernel_size[0] + 2
        stride         = child.stride[0]
        padding        = kernel_size//2

        if not replace_first_conv_by_1x1_h:
            new_layer      = SESConv_Z2_H(in_channels= in_channels, out_channels= out_channels,\
                                      kernel_size= kernel_size, effective_size= effective_size,\
                                      scales= scales, stride= stride, padding= padding, bias= False,\
                                      padding_mode= sesn_padding_mode)

        else:
            # Replace by SESConv_H_H_1x1
            scale_size     = 1
            new_layer      = SESConv_H_H_1x1(in_channels= in_channels, out_channels= out_channels,\
                                          scale_size= scale_size, stride= stride, bias= False)
            # new_layer      = nn.Sequential( SESCopy_Scale(scales= scales), temp_layer)


    elif isinstance(child, nn.Conv2d):
        in_channels    = child.in_channels
        out_channels   = child.out_channels
        effective_size = child.kernel_size[0]
        stride         = child.stride[0] if len(child.stride)> 1 else child.stride

        if effective_size > 1:
            # Replace by SESConv_H_H
            kernel_size    = child.kernel_size[0] + 2
            padding        = kernel_size//2

            new_layer      = SESConv_H_H(in_channels= in_channels, out_channels= out_channels,\
                                         scale_size= 1, kernel_size= kernel_size,\
                                         effective_size= effective_size, scales= scales,\
                                         stride= stride, padding= padding,\
                                         bias= False, padding_mode= sesn_padding_mode)

        else:
            # Replace by SESConv_H_H_1x1
            scale_size     = 1

            new_layer      = SESConv_H_H_1x1(in_channels= in_channels, out_channels= out_channels,\
                                          scale_size= scale_size, stride= stride, bias= False)

    elif isinstance(child, nn.BatchNorm2d):
        # Replace by BatchNorm3d
        num_features        = child.num_features
        eps                 = child.eps
        momentum            = child.momentum
        affine              = child.affine
        track_running_stats = child.track_running_stats
        new_layer           = nn.BatchNorm3d(num_features= num_features, eps= eps, momentum= momentum,\
                                             affine= affine, track_running_stats= track_running_stats)

    elif isinstance(child, nn.MaxPool2d) or isinstance(child, nn.AvgPool2d):
        kernel_size    = (1, child.kernel_size, child.kernel_size)
        stride         = (1, child.stride, child.stride)
        padding        = (0, child.padding, child.padding)
        ceil_mode      = child.ceil_mode
        if isinstance(child, nn.MaxPool2d):
            # Replace by AvgPool3d
            dilation       = child.dilation
            return_indices = child.return_indices
            new_layer      = nn.MaxPool3d(kernel_size= kernel_size, stride= stride, padding= padding,\
                                      dilation= dilation, return_indices= return_indices, ceil_mode= ceil_mode)
        else:
            # Replace by AvgPool3d
            new_layer      = nn.AvgPool3d(kernel_size= kernel_size, stride= stride, padding= padding,\
                                      ceil_mode= ceil_mode)

    if new_layer is not None:
        # Replace the layer
        setattr(network, child_name, new_layer)
        return True
    else:
        return False

def add_SESMax_Scale_to_child(network, child_name):
    # Add a SES_max_scale at the end
    module_list_with_key  = OrderedDict([  ('extra0', getattr(network, child_name)),         ('extra1', SESMax_Scale())          ])
    new_layer             = nn.Sequential(module_list_with_key)
    # new_layer             = nn.Sequential(getattr(network, child_name), SESMax_Scale())
    setattr(network, child_name, new_layer)
    return network

def add_SESCopy_Scale_to_child(network, child_name, scales):
    module_list_with_key  = OrderedDict([  ('extra0', SESCopy_Scale(scales= scales)),         ('extra1', getattr(network, child_name))         ])
    new_layer             = nn.Sequential(module_list_with_key)
    # new_layer             = nn.Sequential(SESCopy_Scale(scales= scales), getattr(network, child_name))
    setattr(network, child_name, new_layer)
    return network

def sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(network, child_name, scales):
    module_list_with_key  = OrderedDict([  ('extra0', SESCopy_Scale(scales= scales)),         ('extra1', getattr(network, child_name)), ('extra2', SESMax_Scale())         ])
    new_layer             = nn.Sequential(module_list_with_key)
    # new_layer             = nn.Sequential(SESCopy_Scale(scales= scales), getattr(network, child_name), SESMax_Scale())
    setattr(network, child_name, new_layer)
    return network

def get_child_names_of_network(network):
    network_child_names = []
    for child_name, child in network.named_children():
        network_child_names.append(child_name)

    return network_child_names

def replace_all_layers_by_sesn_layers(network, scales, replace_style= None, sesn_padding_mode="replicate"):
    # Reference
    # https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/7
    max_scale_before_after_transition       = replace_style == "max_scale_before_after_transition"
    max_scale_after_block                   = replace_style == "max_scale_after_block"
    max_scale_after_block_with_HH           = replace_style == "max_scale_after_block_with_HH"
    max_scale_after_conv_kernel_more_than_1 = replace_style == "max_scale_after_conv_kernel_more_than_1"
    max_scale_after_conv_kernel_all         = replace_style == "max_scale_after_conv_kernel_all"
    max_scale_after_conv_kernel_all_by_ZH   = replace_style == "max_scale_after_conv_kernel_all_by_ZH"
    max_scale_after_dla34_layer             = replace_style == "max_scale_after_dla34_layer"

    valid_block_flag = isinstance(network, BasicBlock) or isinstance(network, _DenseLayer)
    check_flag       = max_scale_after_block and valid_block_flag
    check_flag_HH    = max_scale_after_block_with_HH and valid_block_flag

    check_flag_dla34 = max_scale_after_dla34_layer and valid_block_flag
    first_conv_found = False

    for child_name, child in network.named_children():

        if max_scale_after_conv_kernel_more_than_1:

            if isinstance(child, nn.MaxPool2d) or isinstance(child, nn.AvgPool2d) or isinstance(child, nn.BatchNorm2d):
                # Do not replace anything
                continue

            elif isinstance(child, nn.Conv2d):
                if child.kernel_size[0] > 1:
                    replace_single_layer(network, child, child_name, scales, first_conv= True, sesn_padding_mode= sesn_padding_mode)
                    network = add_SESMax_Scale_to_child(network, child_name)
                else:
                    pass
            else:
                replace_all_layers_by_sesn_layers(child, scales, replace_style= replace_style, sesn_padding_mode= sesn_padding_mode)


        elif max_scale_after_conv_kernel_all:

            if isinstance(child, nn.MaxPool2d) or isinstance(child, nn.AvgPool2d) or isinstance(child, nn.BatchNorm2d):
                # Do not replace anything
                continue

            elif isinstance(child, nn.Conv2d):
                effective_size = child.kernel_size[0]
                if effective_size > 1:
                    replace_first_conv_by_1x1_h = False
                    _ = replace_single_layer(network, child, child_name, scales, first_conv= True, replace_first_conv_by_1x1_h= replace_first_conv_by_1x1_h, sesn_padding_mode= sesn_padding_mode)
                    network        = add_SESMax_Scale_to_child(network, child_name)
                else:
                    replace_first_conv_by_1x1_h = True
                    _ = replace_single_layer(network, child, child_name, scales, first_conv= True, replace_first_conv_by_1x1_h= replace_first_conv_by_1x1_h, sesn_padding_mode= sesn_padding_mode)
                    new_layer      = nn.Sequential( SESCopy_Scale(scales= scales), getattr(network, child_name),  SESMax_Scale())
                    setattr(network, child_name, new_layer)

            else:
                replace_all_layers_by_sesn_layers(child, scales, replace_style= replace_style, sesn_padding_mode= sesn_padding_mode)

        elif max_scale_after_conv_kernel_all_by_ZH:
            if isinstance(child, nn.MaxPool2d) or isinstance(child, nn.AvgPool2d) or isinstance(child, nn.BatchNorm2d):
                # Do not replace anything
                continue

            elif isinstance(child, nn.Conv2d):
                replace_first_conv_by_1x1_h = False
                _ = replace_single_layer(network, child, child_name, scales, first_conv= True, replace_first_conv_by_1x1_h= replace_first_conv_by_1x1_h, sesn_padding_mode= sesn_padding_mode)
                network        = add_SESMax_Scale_to_child(network, child_name)

            else:
                replace_all_layers_by_sesn_layers(child, scales, replace_style= replace_style, sesn_padding_mode= sesn_padding_mode)

        elif max_scale_after_block_with_HH:

            first_conv_flag = False

            if "transition" in child_name or "norm5" in child_name:
                # Do not replace anything for transition blocks
                continue

            if check_flag_HH and not first_conv_found:
                if isinstance(child, nn.MaxPool2d) or isinstance(child, nn.AvgPool2d) or isinstance(child, nn.BatchNorm2d):
                    # If instance of pool or norm before conv has been replaced.
                    # Do not replace anything
                    continue

                if isinstance(child, nn.Conv2d):
                    first_conv_found = True
                    first_conv_flag  = True
            #         effective_size   = child.kernel_size[0]
            #         if effective_size >
            #             replace_first_conv_by_1x1_h=

            if replace_single_layer(network, child, child_name, scales, first_conv= False, sesn_padding_mode= sesn_padding_mode):
                if first_conv_flag:
                    # Add a copy layer
                    temp_layer = getattr(network, child_name)
                    new_layer  = nn.Sequential( SESCopy_Scale(scales= scales), temp_layer)
                    setattr(network, child_name, new_layer)
            else:
                replace_all_layers_by_sesn_layers(child, scales, replace_style= replace_style, sesn_padding_mode= sesn_padding_mode)

        elif max_scale_after_dla34_layer:
            first_conv_flag = False

            if check_flag_dla34 and not first_conv_found:
                if isinstance(child, nn.MaxPool2d) or isinstance(child, nn.AvgPool2d) or isinstance(child, nn.BatchNorm2d):
                    # If instance of pool or norm before conv has been replaced.
                    # Do not replace anything
                    continue

                if isinstance(child, nn.Conv2d):
                    print("first conv found ...")
                    first_conv_found = True
                    first_conv_flag  = True

            if not replace_single_layer(network, child, child_name, scales, first_conv= first_conv_flag, sesn_padding_mode= sesn_padding_mode):
                replace_all_layers_by_sesn_layers(child, scales, replace_style= replace_style, sesn_padding_mode= sesn_padding_mode)

        elif max_scale_before_after_transition:

            if "transition" in child_name:
                # Put an extra scales before and after transition
                new_layer      = nn.Sequential(SESMax_Scale(), getattr(network, child_name),   SESCopy_Scale(scales= scales))
                setattr(network, child_name, new_layer)
                continue

            if "norm5" in child_name:
                # Put SESMax_Scale
                new_layer      = nn.Sequential(SESMax_Scale(), getattr(network, child_name))
                setattr(network, child_name, new_layer)
                continue

            if not replace_single_layer(network, child, child_name, scales, first_conv= False, sesn_padding_mode= sesn_padding_mode):
                replace_all_layers_by_sesn_layers(child, scales, replace_style= replace_style, sesn_padding_mode= sesn_padding_mode)


        else:
            first_conv_flag = False

            if max_scale_after_block and "transition" in child_name or "norm5" in child_name:
                # Do not replace anything for transition blocks
                continue

            if check_flag and not first_conv_found:
                if isinstance(child, nn.MaxPool2d) or isinstance(child, nn.AvgPool2d) or isinstance(child, nn.BatchNorm2d):
                    # If instance of pool or norm before conv has been replaced.
                    # Do not replace anything
                    continue

                if isinstance(child, nn.Conv2d):
                    first_conv_found = True
                    first_conv_flag  = True

            if not replace_single_layer(network, child, child_name, scales, first_conv= first_conv_flag, sesn_padding_mode= sesn_padding_mode):
                replace_all_layers_by_sesn_layers(child, scales, replace_style= replace_style, sesn_padding_mode= sesn_padding_mode)

    if check_flag and first_conv_found:
        # Add a SES_max_scale at the end
        new_layer             = nn.Sequential(getattr(network, child_name), SESMax_Scale())
        setattr(network, child_name, new_layer)

    if check_flag_HH or check_flag_dla34:
        # Add a SES_max_scale at the end
        new_layer             = nn.Sequential(getattr(network, child_name), SESMax_Scale())
        setattr(network, child_name, new_layer)


def add_log_polar_conv(network, do_upsample= True, upsample_factor= 4, debug= False):
    # Reference
    # https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/7
    for child_name, child in network.named_children():
        if isinstance(child, nn.Conv2d):
            if child.kernel_size[0] > 1:
                # Not 1x1 convolution
                setattr(network, child_name, LogPolarConvolution(child, do_upsample= do_upsample, upsample_factor= upsample_factor))
                if debug:
                    print("Found " + child_name)
        else:
            add_log_polar_conv(child, do_upsample= do_upsample, upsample_factor= upsample_factor)

def add_dilated_conv(network, scales, debug= False):
    # Reference
    # https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/7
    for child_name, child in network.named_children():
        if isinstance(child, nn.Conv2d):
            if child.kernel_size[0] > 1:
                # Not 1x1 convolution
                setattr(network, child_name, DilatedConvolution(child, scales))
                if debug:
                    print("Found " + child_name)
        else:
            add_dilated_conv(child, scales)

def replace_and_initialize_with_transform_weights(base, first_child_name, sesn_scales, replace_all= True, replace_layer_names= None, replace_style= None, sesn_padding_mode= "replicate", scale_index_for_init= 0):
    # If we do copy.deepcopy we used to get a key mismatch error with orig --> we no longer copy weights explicitly when layer is not replaced
    # If we do not do copy.deepcopy, some of the src weights are overwritten.
    # Make sure you use copy.DEEPcopy
    dest = copy.deepcopy(base)
    # Get all layer names
    all_layer_names = get_child_names_of_network(network= dest)

    max_scale_after_end                     = replace_style == "max_scale_after_end"
    max_scale_before_after_transition       = replace_style == "max_scale_before_after_transition"
    max_scale_after_block                   = replace_style == "max_scale_after_block"
    max_scale_after_block_with_HH           = replace_style == "max_scale_after_block_with_HH"
    max_scale_after_conv_kernel_more_than_1 = replace_style == "max_scale_after_conv_kernel_more_than_1"
    max_scale_after_conv_kernel_all         = replace_style == "max_scale_after_conv_kernel_all"
    max_scale_after_conv_kernel_all_by_ZH   = replace_style == "max_scale_after_conv_kernel_all_by_ZH"
    max_scale_after_dla34_layer             = replace_style == "max_scale_after_dla34_layer"
    # =====================================================================
    # Decide whether we replace few or all layers
    # =====================================================================
    if replace_all:

        if ('ResNet' in type(base).__name__):
            # ResNet style networks
            for child_name, child in dest.named_children():
                if child_name == "conv1":
                    replace_single_layer(dest, getattr(dest, child_name), child_name, scales= sesn_scales, first_conv= True , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                elif child_name == "bn1":
                    replace_single_layer(dest, getattr(dest, child_name), child_name, scales= sesn_scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                elif "layer" in child_name:
                    replace_single_layer(child[0], getattr(child[0], 'conv1'), 'conv1', scales= sesn_scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(child[0], getattr(child[0], 'bn1')  , 'bn1'  , scales= sesn_scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(child[0], getattr(child[0], 'conv2'), 'conv2', scales= sesn_scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(child[0], getattr(child[0], 'bn2')  , 'bn2'  , scales= sesn_scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    replace_single_layer(child[1], getattr(child[1], 'conv1'), 'conv1', scales= sesn_scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(child[1], getattr(child[1], 'bn1')  , 'bn1'  , scales= sesn_scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(child[1], getattr(child[1], 'conv2'), 'conv2', scales= sesn_scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(child[1], getattr(child[1], 'bn2')  , 'bn2'  , scales= sesn_scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    if child_name == "layer2" or child_name == "layer3" or child_name == "layer4":
                        # We have a downsample block
                        downblock = getattr(child[0], 'downsample')
                        replace_single_layer(downblock, getattr(downblock, '0'), '0', scales= sesn_scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                        replace_single_layer(downblock, getattr(downblock, '1'), '1', scales= sesn_scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

        elif max_scale_after_dla34_layer:
            # Max_Scale at intermediate Tree
            for child_name, child in dest.named_children():
                if 'base_layer' in child_name or 'level0' in child_name or 'level1' in child_name:
                    replace_single_layer(child, getattr(child, '0'), '0', scales= sesn_scales, first_conv= True , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(child, getattr(child, '1'), '1', scales= sesn_scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    add_SESMax_Scale_to_child(dest, child_name)
                    continue

                if 'level2' in child_name or 'level5' in child_name:
                    # Downsample + project
                    replace_single_layer(child    , getattr(child, 'downsample') , 'downsample' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    temp_parent = getattr(child, 'project')
                    replace_single_layer(temp_parent, getattr(temp_parent, '0') , '0' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, '1') , '1' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # tree1
                    temp_parent = getattr(child, 'tree1')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn1'  ), 'bn1'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn2'  ), 'bn2'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # tree2
                    temp_parent = getattr(child, 'tree2')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn1'  ), 'bn1'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn2'  ), 'bn2'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    temp_parent = getattr(child, 'root')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv') , 'conv' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn'  ) , 'bn'   , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(dest, child_name, scales= sesn_scales)
                    continue

                if 'level3' in child_name or 'level4' in child_name:
                    # outer downsample + project
                    replace_single_layer(child    , getattr(child, 'downsample') , 'downsample' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    add_SESCopy_Scale_to_child(child, 'downsample', scales= sesn_scales)

                    temp_parent = getattr(child, 'project')
                    replace_single_layer(temp_parent, getattr(temp_parent, '0') , '0' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, '1') , '1' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    add_SESMax_Scale_to_child(child, 'project')

                    # outer.tree1.downsample + project
                    temp_parent = getattr(child, 'tree1')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'downsample') , 'downsample' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    temp_parent = getattr(getattr(child, 'tree1'), 'project')
                    replace_single_layer(temp_parent, getattr(temp_parent, '0') , '0' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, '1') , '1' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # outer.tree1.tree1
                    temp_parent = getattr(getattr(child, 'tree1'), 'tree1')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn1'  ), 'bn1'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn2'  ), 'bn2'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # outer.tree1.tree2
                    temp_parent = getattr(getattr(child, 'tree1'), 'tree2')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn1'  ), 'bn1'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn2'  ), 'bn2'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # outer.tree1.root
                    temp_parent = getattr(getattr(child, 'tree1'), 'root')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv') , 'conv' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn'  ) , 'bn'   , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(child, 'tree1', scales= sesn_scales)

                    # outer.tree2.tree1
                    temp_parent = getattr(getattr(child, 'tree2'), 'tree1')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False  , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn1'  ), 'bn1'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn2'  ), 'bn2'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # outer.tree2.tree2
                    temp_parent = getattr(getattr(child, 'tree2'), 'tree2')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn1'  ), 'bn1'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn2'  ), 'bn2'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # outer.tree2.root
                    temp_parent = getattr(getattr(child, 'tree2'), 'root')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv') , 'conv' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn'  ) , 'bn'   , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(child, 'tree2', scales= sesn_scales)

        elif max_scale_after_end:
            # Everything in 5D, Max_Scale at the end
            for child_name, child in dest.named_children():
                if 'base_layer' in child_name or 'level0' in child_name or 'level1' in child_name:
                    replace_single_layer(child, getattr(child, '0'), '0', scales= sesn_scales, first_conv= True , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(child, getattr(child, '1'), '1', scales= sesn_scales, first_conv= False, replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    continue

                if 'level2' in child_name or 'level5' in child_name:
                    # Downsample + project
                    replace_single_layer(child    , getattr(child, 'downsample') , 'downsample' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    temp_parent = getattr(child, 'project')
                    replace_single_layer(temp_parent, getattr(temp_parent, '0') , '0' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, '1') , '1' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # tree1
                    temp_parent = getattr(child, 'tree1')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn1'  ), 'bn1'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn2'  ), 'bn2'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # tree2
                    temp_parent = getattr(child, 'tree2')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn1'  ), 'bn1'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn2'  ), 'bn2'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    temp_parent = getattr(child, 'root')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv') , 'conv' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn'  ) , 'bn'   , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(dest, child_name, scales= sesn_scales)
                    continue

                if 'level3' in child_name or 'level4' in child_name:
                    # outer downsample + project
                    replace_single_layer(child    , getattr(child, 'downsample') , 'downsample' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    # add_SESCopy_Scale_to_child(child, 'downsample', scales= sesn_scales)

                    temp_parent = getattr(child, 'project')
                    replace_single_layer(temp_parent, getattr(temp_parent, '0') , '0' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, '1') , '1' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    add_SESMax_Scale_to_child(child, 'project')

                    # outer.tree1.downsample + project
                    temp_parent = getattr(child, 'tree1')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'downsample') , 'downsample' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    temp_parent = getattr(getattr(child, 'tree1'), 'project')
                    replace_single_layer(temp_parent, getattr(temp_parent, '0') , '0' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, '1') , '1' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # outer.tree1.tree1
                    temp_parent = getattr(getattr(child, 'tree1'), 'tree1')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn1'  ), 'bn1'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn2'  ), 'bn2'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # outer.tree1.tree2
                    temp_parent = getattr(getattr(child, 'tree1'), 'tree2')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn1'  ), 'bn1'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn2'  ), 'bn2'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # outer.tree1.root
                    temp_parent = getattr(getattr(child, 'tree1'), 'root')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv') , 'conv' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn'  ) , 'bn'   , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(child, 'tree1', scales= sesn_scales)

                    # outer.tree2.tree1
                    temp_parent = getattr(getattr(child, 'tree2'), 'tree1')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False  , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn1'  ), 'bn1'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn2'  ), 'bn2'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # outer.tree2.tree2
                    temp_parent = getattr(getattr(child, 'tree2'), 'tree2')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn1'  ), 'bn1'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn2'  ), 'bn2'  , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    # outer.tree2.root
                    temp_parent = getattr(getattr(child, 'tree2'), 'root')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv') , 'conv' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'bn'  ) , 'bn'   , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)

                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(child, 'tree2', scales= sesn_scales)

        elif max_scale_after_conv_kernel_all:
            # Only conv process in 5D, Max_Scale immediately after every conv
            for child_name, child in dest.named_children():
                if 'base_layer' in child_name or 'level0' in child_name or 'level1' in child_name:
                    replace_single_layer(child, getattr(child, '0'), '0', scales= sesn_scales, first_conv= True , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    add_SESMax_Scale_to_child(child, '0')
                    continue

                if 'level2' in child_name or 'level5' in child_name:
                    # Downsample + project
                    temp_parent = getattr(child, 'project')
                    replace_single_layer(temp_parent, getattr(temp_parent, '0') , '0' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, '0', scales= sesn_scales)

                    # tree1
                    temp_parent = getattr(child, 'tree1')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv1', scales= sesn_scales)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv2', scales= sesn_scales)

                    # tree2
                    temp_parent = getattr(child, 'tree2')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv1', scales= sesn_scales)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv2', scales= sesn_scales)

                    temp_parent = getattr(child, 'root')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv') , 'conv' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv', scales= sesn_scales)

                    continue

                if 'level3' in child_name or 'level4' in child_name:
                    # outer downsample + project
                    temp_parent = getattr(child, 'project')
                    replace_single_layer(temp_parent, getattr(temp_parent, '0') , '0' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, '0', scales= sesn_scales)

                    # outer.tree1.downsample + project
                    temp_parent = getattr(getattr(child, 'tree1'), 'project')
                    replace_single_layer(temp_parent, getattr(temp_parent, '0') , '0' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, '0', scales= sesn_scales)

                    # outer.tree1.tree1
                    temp_parent = getattr(getattr(child, 'tree1'), 'tree1')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv1', scales= sesn_scales)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv2', scales= sesn_scales)

                    # outer.tree1.tree2
                    temp_parent = getattr(getattr(child, 'tree1'), 'tree2')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv1', scales= sesn_scales)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv2', scales= sesn_scales)

                    # outer.tree1.root
                    temp_parent = getattr(getattr(child, 'tree1'), 'root')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv') , 'conv' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv', scales= sesn_scales)

                    # outer.tree2.tree1
                    temp_parent = getattr(getattr(child, 'tree2'), 'tree1')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False  , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv1', scales= sesn_scales)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv2', scales= sesn_scales)

                    # outer.tree2.tree2
                    temp_parent = getattr(getattr(child, 'tree2'), 'tree2')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv1'), 'conv1', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv1', scales= sesn_scales)
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv2'), 'conv2', scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv2', scales= sesn_scales)

                    # outer.tree2.root
                    temp_parent = getattr(getattr(child, 'tree2'), 'root')
                    replace_single_layer(temp_parent, getattr(temp_parent, 'conv') , 'conv' , scales= sesn_scales, first_conv= False , replace_first_conv_by_1x1_h= False, sesn_padding_mode= sesn_padding_mode)
                    sandwich_layer_between_SESCopy_Scale_and_SESMax_Scale(temp_parent, 'conv', scales= sesn_scales)


        # # Replace the first layer by SS_ZH
        # _ = replace_single_layer(network= dest, child= getattr(dest, first_child_name), child_name= first_child_name, scales= sesn_scales, first_conv= True, sesn_padding_mode= sesn_padding_mode)
        #
        # if 'pool0' in all_layer_names:
        #     pool_layer = 'pool0'
        #     norm_layer = 'norm0'
        # elif 'pool1' in all_layer_names:
        #     pool_layer = 'pool1'
        #     norm_layer = 'norm1'
        #
        # if max_scale_after_conv_kernel_more_than_1 or max_scale_after_conv_kernel_all or max_scale_after_conv_kernel_all_by_ZH:
        #     dest = add_SESMax_Scale_to_child(dest, first_child_name)
        #
        # elif max_scale_after_block or max_scale_after_block_with_HH:
        #     new_layer             = nn.Sequential(getattr(dest, pool_layer), SESMax_Scale())
        #     setattr(dest, pool_layer, new_layer)

        # Replace other layers
        # replace_all_layers_by_sesn_layers(dest, scales= sesn_scales, replace_style= replace_style, sesn_padding_mode= sesn_padding_mode)

        # if max_scale_before_after_transition:
        #     new_layer             = nn.Sequential(getattr(dest, pool_layer), SESMax_Scale(), SESCopy_Scale(scales= sesn_scales))
        #     setattr(dest, pool_layer, new_layer)
        #
        # if max_scale_after_end and 'norm5' in all_layer_names:
        #     # Replace the batch norm at the end
        #      _ = replace_single_layer(network= dest, child= getattr(dest, "norm5"), child_name= "norm5", scales= sesn_scales, first_conv= False, sesn_padding_mode= sesn_padding_mode)

    else:
        logging.info ("Replacing few layers --> " + ','.join(replace_layer_names))
        num_replace_layers  = len(replace_layer_names)

        if max_scale_after_conv_kernel_all or max_scale_after_conv_kernel_all_by_ZH:

            # Replace other layers
            for i in range(num_replace_layers):
                curr_layer_name            = replace_layer_names[i]

                if curr_layer_name not in all_layer_names:
                    logging.info("WARNING! replace_layer_name {} not present in the network. Exiting!!!".format(curr_layer_name))
                    continue

                if curr_layer_name == first_child_name:
                    # Replace the first layer by SS_ZH
                    _ = replace_single_layer(network= dest, child= getattr(dest, first_child_name), child_name= first_child_name, scales= sesn_scales, first_conv= True, sesn_padding_mode= sesn_padding_mode)

                    if max_scale_after_conv_kernel_more_than_1 or max_scale_after_conv_kernel_all:
                        dest = add_SESMax_Scale_to_child(dest, first_child_name)

                # Replace other layers
                replace_all_layers_by_sesn_layers(getattr(dest, curr_layer_name), scales= sesn_scales, replace_style= replace_style, sesn_padding_mode= sesn_padding_mode)

        else:
            # Replace other layers
            for i in range(num_replace_layers):
                curr_layer_name            = replace_layer_names[i]

                if curr_layer_name not in all_layer_names:
                    logging.info("WARNING! replace_layer_name {} not present in the network. Exiting!!!".format(curr_layer_name))
                    continue
                curr_layer_index_in_object = all_layer_names.index(curr_layer_name)

                if max_scale_after_block or max_scale_after_block_with_HH:
                    if "transition" in curr_layer_name or "norm5" in curr_layer_name:
                        # Do not replace anything for transition blocks
                        continue

                if not (max_scale_after_block or max_scale_after_block_with_HH):
                    if i == 0 and replace_layer_names[i] != all_layer_names[0]:
                        # Add an extra SESCopy_Scale before the first layer
                        prev_layer_name     = all_layer_names        [curr_layer_index_in_object - 1]
                        new_conv            = SESCopy_Scale(scales= sesn_scales)

                        layer_list_with_key = OrderedDict([('orig', getattr(dest, prev_layer_name)), ('extra', new_conv)])
                        new_layer           = nn.Sequential(layer_list_with_key)
                        setattr(dest, prev_layer_name, new_layer)

                if curr_layer_name == first_child_name:
                    # Replace the first layer by SS_ZH
                    _ = replace_single_layer(network= dest, child= getattr(dest, first_child_name), child_name= first_child_name, scales= sesn_scales, first_conv= True, sesn_padding_mode= sesn_padding_mode)

                elif curr_layer_name == "norm0" or curr_layer_name == "norm1" or curr_layer_name == "norm5":
                    # Instance of single layer
                    _ = replace_single_layer(network= dest, child= getattr(dest, curr_layer_name), child_name= curr_layer_name, scales= sesn_scales, sesn_padding_mode= sesn_padding_mode)

                elif 'pool0' in curr_layer_name or 'pool1' in curr_layer_name:
                    if 'pool0' in curr_layer_name:
                        pool_layer = 'pool0'
                    elif 'pool1' in curr_layer_name:
                        pool_layer = 'pool1'

                    # Instance of single layer
                    _ = replace_single_layer(network= dest, child= getattr(dest, curr_layer_name), child_name= curr_layer_name, scales= sesn_scales, sesn_padding_mode= sesn_padding_mode)

                    if max_scale_after_block or max_scale_after_block_with_HH:
                        new_layer             = nn.Sequential(getattr(dest, pool_layer), SESMax_Scale())
                        setattr(dest, pool_layer, new_layer)

                else:
                    replace_all_layers_by_sesn_layers(getattr(dest, curr_layer_name), scales= sesn_scales, replace_style= replace_style, sesn_padding_mode= sesn_padding_mode)

    for m in base.modules():
        if isinstance(m, (SESConv_H_H, SESConv_Z2_H, SESConv_H_H_1x1)):
            n = m.weight.nelement() / m.in_channels
            m.weight.data.normal_(0, (2 / n)**0.5)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # =====================================================================
    # Transform old to new weights and add SESMax_Scale at the end
    # =====================================================================
    # Init all scales with imageNet weights
    for index_for_init in range(len(sesn_scales)):
        dest_state_dict = calculate_new_weights(src_input= base, dest_input= dest, replace_all= replace_all,\
                                                replace_layer_names= replace_layer_names, replace_style= replace_style,\
                                                scale_index_for_init= index_for_init)
        dest.load_state_dict(dest_state_dict)



    if max_scale_before_after_transition or max_scale_after_block or max_scale_after_block_with_HH \
            or max_scale_after_conv_kernel_more_than_1 or max_scale_after_conv_kernel_all or \
            max_scale_after_dla34_layer or max_scale_after_conv_kernel_all_by_ZH:
        # They already have added the SESMax_Scale() block. Do nothing
        final_base = dest
    else:
        final_base = nn.Sequential(dest, SESMax_Scale())
    print(final_base)
    return final_base
