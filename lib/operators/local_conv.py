import torch
import torch.nn as nn
import torch.nn.functional as F

'''
References: 
1. Monocular 3D Region Proposal Network for Object Detection, ICCV'19
2. tensor.unfold
        Returns a view of the original tensor which contains all slices of size size from self tensor in the dimension dimension.
        Step between two slices is given by step.
        If sizedim is the size of dimension dimension for self, the size of dimension dimension in the returned tensor will be (sizedim - size) / step + 1.
        An additional dimension of size size is appended in the returned tensor.
        pytorch doc: https://pytorch.org/docs/stable/tensors.html?highlight=unfold#torch.Tensor.unfold
'''


class LocalConv2d(nn.Module):

    def __init__(self, num_rows, num_feats_in, num_feats_out, kernel=1, padding=0):
        super(LocalConv2d, self).__init__()

        self.num_rows = num_rows
        self.out_channels = num_feats_out
        self.kernel = kernel
        self.padding = padding
        self.group_conv = nn.Conv2d(num_feats_in * num_rows, num_feats_out * num_rows, kernel, stride=1, groups=num_rows)


    def forward(self, x):
        batch, channel, height, width = x.size()
        if self.pad: x = F.pad(x, [self.padding, self.padding, self.padding, self.padding], mode='constant', value=0)
        bins = int(height / self.num_rows)

        x = x.unfold(2, bins + self.pad * 2, bins)
        x = x.permute([0, 2, 1, 4, 3]).contiguous()
        x = x.view(batch, channel * self.num_rows, bins + self.pad * 2, (width + self.pad * 2)).contiguous()

        # group convolution for efficient parallel processing
        y = self.group_conv(x)
        y = y.view(batch, self.num_rows, self.out_channels, bins, width).contiguous()
        y = y.permute([0, 2, 1, 3, 4]).contiguous()
        y = y.view(batch, self.out_channels, height, width)

        return y


if __name__ == '__main__':
    pass