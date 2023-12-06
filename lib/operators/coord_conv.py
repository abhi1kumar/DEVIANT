import torch
import torch.nn as nn

"""
References:
    An intriguing failing of convolutional neural networks and the CoordConv solution, Neurips'18
    https://arxiv.org/pdf/1807.03247.pdf)
"""

class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, bias=False, with_radius=False):
        super().__init__()
        in_channels = in_channels + 2
        if with_radius:
            in_channels = in_channels + 1
        self.addcoords = AddCoords(with_radius=with_radius)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              stride=stride,
                              kernel_size=kernal_size,
                              padding=kernal_size//2,
                              bias=bias)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class AddCoords(nn.Module):
    def __init__(self, with_radius=False):
        super().__init__()
        self.with_radius = with_radius

    def forward(self, features):
        batch_size, _, x_dim, y_dim = features.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        features_updated = torch.cat([features,
                         xx_channel.type_as(features),
                         yy_channel.type_as(features)], dim=1)

        if self.with_radius:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(features) - 0.5, 2) + torch.pow(yy_channel.type_as(features) - 0.5, 2))
            features_updated = torch.cat([features_updated, rr], dim=1)

        return features_updated




if __name__ == '__main__':
    input = torch.randn(2, 3, 32, 32)
    coordconv = CoordConv2d(in_channels=3, out_channels=6)
    print(coordconv(input).shape)
