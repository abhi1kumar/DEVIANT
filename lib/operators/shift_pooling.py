import torch
import torch.nn as nn

"""
References:
    Learning Depth-Guided Convolutions for Monocular 3D Object Detection, CVPR'2020
"""


def shift_pooling(features, shift_times=3):
    updated_features = features.clone()
    for i in range(1, shift_times):
        updated_features += torch.cat([features[:, i:, :, :], features[:, :i, :, :]], dim=1)
    updated_features /= shift_times
    return updated_features


if __name__ == '__main__':
    f1 = torch.ones(1, 1, 3, 3)
    f2 = f1+1
    f3 = f2+1
    feature = torch.cat([f1, f2, f3], dim=1)
    print(feature)

    print(shift_pooling(feature, 3))