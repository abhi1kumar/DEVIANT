'''
ResNet, https://arxiv.org/abs/1512.03385
'''

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from lib.projective.ses_conv import ses_max_projection


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes,
                               out_channels=planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,  planes, kernel_size=3, stride=stride, padding=1, bias=False)   # key convolution
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, **kwargs):
        super().__init__()
        if num_blocks == [2,2,2,2]:
            # ResNet 18
            self.channels = [32, 32, 64, 128, 256]
        elif num_blocks == [3,4,6,3]:
            # ResNet34/ResNet50
            self.channels = [32, 128, 256, 512, 1024]
        self.return_levels = True if 'return_levels' not in kwargs else kwargs['return_levels']
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block,  64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)

        # Decrease channels of output feature map by 2 to bring in par with DLA34
        self.dec_ch0   = nn.Conv2d(64 , 32 , kernel_size=1, bias=False)
        self.dec_ch1   = nn.Conv2d(64 , 32 , kernel_size=1, bias=False)
        self.dec_ch2   = nn.Conv2d(128, 64 , kernel_size=1, bias=False)
        self.dec_ch3   = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.dec_ch4   = nn.Conv2d(512, 256, kernel_size=1, bias=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_planes,
                                                 planes * block.expansion,
                                                 kernel_size=1,
                                                 stride=stride,
                                                 bias=False),
                                       nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def append_to_output(self, y, out, dec_ch= None):
        if out.ndim > 4:
            temp_out = ses_max_projection(out)
        else:
            temp_out = out

        # If you have to decrease channels
        if dec_ch is not None:
            temp_out = dec_ch(temp_out)

        y.append(temp_out)
        return y

    def forward(self, x):
        y = []
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.return_levels:
            y = self.append_to_output(y, out, dec_ch= self.dec_ch0)

        out = self.layer1(out)
        if self.return_levels:
            y = self.append_to_output(y, out, dec_ch= self.dec_ch1)

        out = self.layer2(out)
        if self.return_levels:
            y = self.append_to_output(y, out, dec_ch= self.dec_ch2)

        out = self.layer3(out)
        if self.return_levels:
            y = self.append_to_output(y, out, dec_ch= self.dec_ch3)

        out = self.layer4(out)
        if self.return_levels:
            y = self.append_to_output(y, out, dec_ch= self.dec_ch4)
            return y
        else:
            return out


def resnet18(pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model.
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        print('===> loading imagenet pretrained model.')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """
    Constructs a ResNet-34 model.
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print('===> loading imagenet pretrained model.')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """
    Constructs a ResNet-50 model.
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print('===> loading imagenet pretrained model.')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """
    Constructs a ResNet-101 model.
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        print('===> loading imagenet pretrained model.')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """
    Constructs a ResNet-152 model.
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        print('===> loading imagenet pretrained model.')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model



if __name__ == '__main__':
    import torch
    net = resnet50(pretrained=True)
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())