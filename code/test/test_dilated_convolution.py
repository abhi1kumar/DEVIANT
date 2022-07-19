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

from lib.projective.projective_utils import DilatedConvolution

a = torch.arange(25).reshape(1,1,5,5).float()

init_conv = nn.Conv2d(in_channels= 1, out_channels= 2, kernel_size= 3, stride= 1, padding= 1, bias= False)
init_conv.weight = nn.Parameter(torch.ones(2,1,3,3))

print(init_conv(a))

d = DilatedConvolution(init_conv, scales= [0.25, 0.5, 1.0])

print(d(a))