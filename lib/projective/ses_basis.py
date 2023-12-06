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


import numpy as np
import torch
import torch.nn.functional as F
import math
from scipy.special import ive

EPS = 1e-3
def hermite_poly(X, n):
    """Hermite polynomial of order n calculated at X
    Args:
        n: int >= 0
        X: np.array

    Output:
        Y: array of shape X.shape
    """
    coeff = [0] * n + [1]
    func = np.polynomial.hermite_e.hermeval(X, coeff)
    return func


def onescale_grid_hermite_gaussian(size, scale, max_order=None):
    max_order = max_order or size - 1
    X = np.linspace(-(size // 2), size // 2, size)
    Y = np.linspace(-(size // 2), size // 2, size)
    order_y, order_x = np.indices([max_order + 1, max_order + 1])

    G = np.exp(-X**2 / (2 * scale**2)) / scale

    basis_x = [G * hermite_poly(X / scale, n) for n in order_x.ravel()]
    basis_y = [G * hermite_poly(Y / scale, n) for n in order_y.ravel()]
    basis_x = torch.Tensor(np.stack(basis_x))
    basis_y = torch.Tensor(np.stack(basis_y))
    basis = torch.bmm(basis_x[:, :, None], basis_y[:, None, :])
    return basis


def steerable_A(size, scales, effective_size, **kwargs):
    max_order = effective_size - 1
    max_scale = max(scales)
    basis_tensors = []
    for scale in scales:
        size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
        basis = onescale_grid_hermite_gaussian(size_before_pad, scale, max_order)
        basis = basis[None, :, :, :]
        pad_size = (size - size_before_pad) // 2
        basis = F.pad(basis, [pad_size] * 4)[0]
        basis_tensors.append(basis)
    return torch.stack(basis_tensors, 1)


def rescale_basis_for_satisfying_constraints(basis):

    if basis.shape[2] == 5:
        basis[2, 1] *= 0.60 #0.61 #0.340
        basis[2, 2] *= 0.45 #0.45 #0.252

        basis[6, 1] *= 0.60 #0.600 #0.536 #0.600
        basis[6, 2] *= 0.45 #0.449 #0.500 #0.449

        basis[8, 1] *= 1.21
        basis[8, 2] *= 0.61

    elif basis.shape[2] == 9:
        basis[0, 1] *= 0.844
        basis[0, 2] *= 0.705

        basis[2, 1] *= 1.16
        basis[2, 2] *= 1.38

        basis[4, 1] *= -0.29
        basis[4, 2] *= -0.21

        basis[6, 1] *= 5.29
        basis[6, 2] *= 1.08

        basis[14, 1] *= 1.16
        basis[14, 2] *= 1.38

        basis[16, 1] *= 0.42
        basis[16, 2] *= 0.006

        basis[18, 1] *= 3.71
        basis[18, 2] *= 1.11

        basis[20, 1] *= 32.64
        basis[20, 2] *= 0.98

        basis[28, 1] *= -0.29
        basis[28, 2] *= -0.21

        basis[30, 1] *= 3.71
        basis[30, 2] *= 1.11

        basis[32, 1] *= 0.33
        basis[32, 2] *= 0.08

        basis[34, 1] *= 0.89
        basis[34, 2] *= 19.59

        basis[42, 1] *= 5.29
        basis[42, 2] *= 1.08

        basis[44, 1] *= -32.64
        basis[44, 2] *= 0.98

        basis[46, 1] *= 0.89
        basis[46, 2] *= 19.59

        basis[48, 1] *= 0.87
        basis[48, 2] *= -0.06

    return basis

def normalize_basis_by_min_scale(basis, norm_per_scale= False, rescale_basis= False):
    norm = basis.pow(2).sum([2, 3], keepdim=True).sqrt()
    if norm_per_scale:
        norm = norm.repeat(1, 1, basis.shape[2], basis.shape[3])
    else:
        norm = norm[:, [0]]
    output = basis.clone()

    if norm_per_scale:
        index  = norm > EPS
    else:
        index  = norm.flatten() > EPS
    output[index] = basis[index] / norm[index]

    if rescale_basis:
        output = rescale_basis_for_satisfying_constraints(output)

    return output
