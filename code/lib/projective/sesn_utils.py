"""
MIT License
Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev

Scale Equivariance Improves Siamese Tracking
Ivan Sosnovik, Artem Moskalev, Arnold Smeulders, WACV 2021
Paper: https://arxiv.org/pdf/2007.09115.pdf
Code: https://github.com/ISosnovik/SiamSE/blob/master/transfer_weights.py

# This script is used to transfer the weights from a pretrained standard model to its scale-equivariant counterpart.
"""
import os, sys
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
import copy
from lib.projective.ses_conv import *
import logging

def tikhonov_reg_lstsq(A, B, eps=1e-12):
    '''|A x - B| + |Gx| -> min_x
    '''
    W = A.shape[1]
    A_inv = np.linalg.inv(A.T @ A + eps * np.eye(W)) @ A.T
    return A_inv @ B


def copy_state_dict_bn(dict_target, dict_origin, key_target, key_origin):
    for postfix in ['weight', 'bias', 'running_mean', 'running_var']:
        dict_target[key_target + '.' + postfix] = dict_origin[key_origin + '.' + postfix]


def copy_state_dict_conv_hh_1x1(dict_target, dict_origin, key_target, key_origin):
    # print(key_origin + '.weight', torch.norm(dict_origin[key_origin + '.weight']))
    # print(key_origin + '.weight', torch.norm(dict_origin[key_origin + '.weight']), torch.norm(dict_target[key_target + '.weight']))
    dict_target[key_target + '.weight'] = dict_origin[key_origin + '.weight']
    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']


def copy_state_dict_conv_hh_1x1_interscale(dict_target, dict_origin, key_target, key_origin):
    w_original = dict_target[key_target + '.weight']
    w_original *= 0
    w_original[:, :, 0] = dict_origin[key_origin + '.weight']
    dict_target[key_target + '.weight'] = w_original
    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']

def get_tensor_shape_as_string(tensor):
    shape_int_list = list(tensor.size())
    shape_str_list = [str(int) for int in shape_int_list]
    return "[" + ','.join(shape_str_list) + "]"

def copy_state_dict_conv_zh(dict_target, dict_origin, key_target, key_origin, scale=0, eps=1e-12):
    weight = dict_origin[key_origin + '.weight']
    basis = dict_target[key_target + '.basis'][:, scale]
    logging.info("Src weight= {}, basis= {}, target weight= {} ".format(get_tensor_shape_as_string(weight),\
                                                                        get_tensor_shape_as_string(basis),\
                                                                        get_tensor_shape_as_string(dict_target[key_target + '.weight'])))

    dict_target[key_target + '.weight'] = _approximate_weight(basis, weight, eps)

    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']


def copy_state_dict_conv_hh(dict_target, dict_origin, key_target, key_origin, scale=0, eps=1e-12):
    weight = dict_origin[key_origin + '.weight']
    basis = dict_target[key_target + '.basis'][:, scale]
    logging.info("Src weight= {}, basis= {}, target weight= {} ".format(get_tensor_shape_as_string(weight),\
                                                                        get_tensor_shape_as_string(basis),\
                                                                        get_tensor_shape_as_string(dict_target[key_target + '.weight'])))
    # No need of weights symmetric
    # after copy.deepcopy in replace_and_initialize_with_transform_weights() of lib/projective_utils
    weight_new = weight #make_weight_symmetric(weight)

    x = torch.zeros_like(dict_target[key_target + '.weight'])
    x[:, :, 0] = _approximate_weight(basis, weight_new, eps)

    dict_target[key_target + '.weight'] = x

    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']

def make_weight_symmetric(weight):
    """
    Some of the weight matrix specially in densenet121 are of shape c_out x c_in x 1 x hw
    This creates problems. Make them into c_out x c_in x h x w
    """
    c_out, c_in, h, w = weight.shape

    weight_sym = weight.clone()
    if h != w:
        if h == 1 or w == 1:
            new_h = int(np.sqrt(h*w))
            weight_sym = weight.reshape(c_out, c_in, new_h, new_h)

    return weight_sym

def copy_state_dict_conv_2d(dict_target, dict_origin, key_target, key_origin):
    dict_target[key_target + '.weight'] = dict_origin[key_origin + '.weight']

    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']

def _approximate_weight(basis, target_weight, eps=1e-12):
    C_out, C_in, h, w = target_weight.shape
    B, H, W = basis.shape
    with torch.no_grad():
        basis = F.pad(basis, [(w - W) // 2, (w - W) // 2, (h - W) // 2, (h - H) // 2])
        target_weight = target_weight.view(C_out * C_in, h * w).detach().cpu().numpy()
        basis = basis.reshape(B, h * w).detach().cpu().numpy()

    assert basis.shape[0] == basis.shape[1]

    matrix_rank = np.linalg.matrix_rank(basis)

    if matrix_rank == basis.shape[0]:
        x = np.linalg.solve(basis.T, target_weight.T).T
    else:
        print('  !!! basis is incomplete. rank={} < num_funcs={}. '
              'weights are approximateb by using '
              'tikhonov regularization'.format(matrix_rank, basis.shape[0]))
        x = tikhonov_reg_lstsq(basis.T, target_weight.T, eps=eps).T

    diff = np.linalg.norm(x @ basis - target_weight)
    norm = np.linalg.norm(target_weight) + 1e-12
    print('  relative_diff={:.1e}, abs_diff={:.1e}'.format(diff / norm, diff))
    x = torch.Tensor(x)
    x = x.view(C_out, C_in, B)
    return x

def convert_param_name_to_layer(name):
    layer_name = '.'.join(name.split('.')[:-1])
    if 'bn' in name or "norm" in name:
        return layer_name, 'bn'
    if 'conv' in name:
        return layer_name, 'conv'
    if 'downsample.0' in name:
        return layer_name, 'conv'
    if 'downsample.1' in name:
        return layer_name, 'bn'

    # Some special resnet18 layers
    if 'dec_ch' in name:
        return layer_name, 'save'

    # Some special dla34 layers
    name = name.replace(".extra0","").replace(".extra1","").replace(".extra","")

    if 'base_layer.0' in name:
        return layer_name, 'conv'
    if 'base_layer.1' in name:
        return layer_name, 'bn'

    if 'level0.0' in name:
        return layer_name, 'conv'
    if 'level0.1' in name:
        return layer_name, 'bn'

    if 'level1.0' in name:
        return layer_name, 'conv'
    if 'level1.1' in name:
        return layer_name, 'bn'

    if 'project.0' in name:
        return layer_name, 'conv'
    if 'project.1' in name:
        return layer_name, 'bn'

    if 'fc' in name:
        return layer_name, 'save'

    if name == 'features.mean' or name == 'features.std':
        return layer_name, 'save'
    print(name)
    raise NotImplementedError

def unique_without_order_change(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def get_layers_name_type_replace_flag(dest_input, replace_all= True, replace_layer_names= None, replace_style= None):
    """
        Returns a list of tuples of the network state dict containing the layer name, type and
        replace_flag (whether the layer got replaced or not).
    """
    max_scale_before_after_transition       = replace_style == "max_scale_before_after_transition"
    max_scale_after_block                   = replace_style == "max_scale_after_block"
    max_scale_after_block_with_HH           = replace_style == "max_scale_after_block_with_HH"
    max_scale_after_conv_kernel_more_than_1 = replace_style == "max_scale_after_conv_kernel_more_than_1"
    max_scale_after_conv_kernel_all         = replace_style == "max_scale_after_conv_kernel_all"
    max_scale_after_conv_kernel_all_by_ZH   = replace_style == "max_scale_after_conv_kernel_all_by_ZH"

    # Convert to eval mode
    dest             = copy.deepcopy(dest_input).eval()
    dest_state_dict  = dest.state_dict()
    dest_keys        = list(dest_state_dict.keys())
    dest_layers_repr = [convert_param_name_to_layer(name) for name in dest_keys]
    dest_layers_repr = list(unique_without_order_change(dest_layers_repr))

    # Get the flags if the layer is getting replaced or not
    dest_layers_name_type_replace_flag = []

    for layer_name, layer_type in dest_layers_repr:
        if max_scale_before_after_transition:
            if "transition" in layer_name or "norm5" in layer_name:
                layer_replace_flag = False
            elif replace_all:
                layer_replace_flag = True
            elif any(s in layer_name for s in replace_layer_names):
                layer_replace_flag = True
            else:
                layer_replace_flag = False

        elif max_scale_after_block:
            if "transition" in layer_name or "norm5" in layer_name or "conv1" in layer_name or "norm1" in layer_name:
                layer_replace_flag = False
            elif replace_all:
                layer_replace_flag = True
            else:
                if any(s in layer_name for s in replace_layer_names):
                    layer_replace_flag = True
                else:
                    layer_replace_flag = False

        elif max_scale_after_block_with_HH:
            if "transition" in layer_name or "norm5" in layer_name or "norm1" in layer_name:
                layer_replace_flag = False
            elif replace_all:
                layer_replace_flag = True
            else:
                if any(s in layer_name for s in replace_layer_names):
                    layer_replace_flag = True
                else:
                    layer_replace_flag = False

        elif max_scale_after_conv_kernel_more_than_1:
            if "transition" in layer_name or "norm" in layer_name or "conv1" in layer_name:
                # conv1 is not replaced in Densenet because it is 1x1 conv
                layer_replace_flag = False
            elif replace_all:
                layer_replace_flag = True
            elif any(s in layer_name for s in replace_layer_names):
                layer_replace_flag = True
            else:
                layer_replace_flag = False

        elif max_scale_after_conv_kernel_all:
            if "transition" in layer_name or "norm" in layer_name:
                # conv1 (1x1 conv) is also replaced in Densenet
                layer_replace_flag = False
            elif replace_all:
                layer_replace_flag = True
            elif any(s in layer_name for s in replace_layer_names):
                layer_replace_flag = True
            else:
                layer_replace_flag = False
        elif max_scale_after_conv_kernel_all_by_ZH:
            if "transition" in layer_name or "norm" in layer_name or "conv1" in layer_name:
                # conv1 of Densenet is replaced by z3H but we can not do weight assignment by 1x1 filter.
                # Hence, set it to false.
                layer_replace_flag = False
            elif replace_all:
                layer_replace_flag = True
            elif any(s in layer_name for s in replace_layer_names):
                layer_replace_flag = True
            else:
                layer_replace_flag = False
        else:
            if replace_all:
                if 'dec_ch' in layer_name:
                    layer_replace_flag = False
                else:
                    layer_replace_flag = True
            else:
                if any(s in layer_name for s in replace_layer_names):
                    layer_replace_flag = True
                else:
                    layer_replace_flag = False

        dest_layers_name_type_replace_flag.append((layer_name, layer_type, layer_replace_flag))

    # for i, (layer_name, layer_type) in enumerate(dest_layers_repr):
    #     print("{:37s} {:5s} {}".format(layer_name, layer_type, replace_flag_for_layers[i]))
    return dest_layers_name_type_replace_flag

def calculate_new_weights(src_input, dest_input, replace_all= True, replace_layer_names= None, replace_style= None, scale_index_for_init= 0):
    """
       dest_input is assumed to be copied of src_input
    """
    max_scale_after_block                   = replace_style == "max_scale_after_block"
    max_scale_after_conv_kernel_more_than_1 = replace_style == "max_scale_after_conv_kernel_more_than_1"

    src  = copy.deepcopy(src_input).eval()
    dest = copy.deepcopy(dest_input).eval()

    src_state_dict = src.state_dict()
    dest_state_dict = dest.state_dict()

    dest_keys        = list(dest_state_dict.keys())
    dest_layers      = list(set(['.'.join(key.split('.')[:-1]) for key in dest_keys]))
    dest_layers_repr = [convert_param_name_to_layer(name) for name in dest_keys]
    dest_layers_repr = list(unique_without_order_change(dest_layers_repr))

    src_keys         = list(src_state_dict.keys())
    src_layer_repr   = [convert_param_name_to_layer(name) for name in src_keys]
    src_layer_names  = [x for x,_ in src_layer_repr]

    dest_layers_name_type_replace_flag = get_layers_name_type_replace_flag(dest_input,\
                                                                           replace_all= replace_all,\
                                                                           replace_layer_names= replace_layer_names,\
                                                                           replace_style= replace_style)

    for i, (layer_name, layer_type, layer_replace_flag) in enumerate(dest_layers_name_type_replace_flag):
        # print('Layer/Type: {}/{}'.format(layer_name, layer_type))
        # print(dest[layer_name].weight.shape)
        # pass

        if not layer_replace_flag:
            pass
            logging.info("{:40s} {:10s} copy src_input weights      ".format(layer_name, layer_type))
            # We do not need to copy the dest explicitly because the dest_input is assumed to be copy of
            # src_input and therefore, has the exact same weights.
        else:
            logging.info("{:40s} {:10s} transform src_input weights ".format(layer_name, layer_type))

            dest_key = layer_name
            src_key  = layer_name
            # Check if the src_keys are intact
            if src_key not in src_layer_names:
                for src_key_curr in src_layer_names:
                    # Check if '.0' is there or '.extra' is there
                    if src_key_curr == dest_key.replace(".extra0", "")\
                            or src_key_curr == dest_key.replace(".extra1", "")\
                            or src_key_curr == dest_key.replace(".0", "")\
                            or src_key_curr == dest_key.replace(".1", "")\
                            or src_key_curr == dest_key.replace(".extra", ""):
                        src_key = src_key_curr
                        break

            if layer_type == 'bn':
                copy_state_dict_bn(dest_state_dict, src_state_dict, key_target= dest_key, key_origin= src_key)

            if layer_type == 'conv':
                weight = dest_state_dict[layer_name + '.weight']

                if layer_replace_flag:
                    # layer replaced. calculate weights by tikhonov regularization

                    if weight.shape[-1] == weight.shape[-2] == 1:
                        if len(weight.shape) == 4:
                            copy_state_dict_conv_hh_1x1(dest_state_dict, src_state_dict, key_target= dest_key, key_origin= src_key)
                        elif len(weight.shape) == 5:
                            copy_state_dict_conv_hh_1x1_interscale(
                                dest_state_dict, src_state_dict, key_target= dest_key, key_origin= src_key)
                        else:
                            raise NotImplementedError
                    elif len(weight.shape) == 4:
                        copy_state_dict_conv_hh(dest_state_dict, src_state_dict, key_target= dest_key, key_origin= src_key, scale= scale_index_for_init)
                    else:
                        copy_state_dict_conv_zh(dest_state_dict, src_state_dict, key_target= dest_key, key_origin= src_key, scale= scale_index_for_init)
                else:
                    # layer not replaced. Use original weights
                    copy_state_dict_conv_2d(dest_state_dict, src_state_dict, dest_key, src_key)

        if layer_type == 'save':
            pass

    return dest_state_dict