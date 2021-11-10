from copy import deepcopy
import torch
from torch import nn
from . import ConvBlock2d

class InvertedResidual2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, expand_ratio = 1):
        super(InvertedResidual2d, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pw
            # layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size = 1))
            layers.append(ConvBlock2d(in_channels, hidden_dim, kernel_size = 1, padding = 0))

        layers.extend([
            # dw
            ConvBlock2d(in_channels, hidden_dim, kernel_size = 3, padding = 1, stride = stride, groups = hidden_dim),
            # ConvBNReLU(hidden_dim, hidden_dim, stride = stride, groups = hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)