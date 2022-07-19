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

import glob

folder = "data/waymo/training/label"

file_names_list = sorted(glob.glob(folder + "/*.txt"))

for temp in file_names_list:
    destination = os.path.realpath(temp)
    if not os.path.exists(destination):
        # print(temp, destination)
        os.system("touch " + destination)