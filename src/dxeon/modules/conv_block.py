from copy import deepcopy
from typing import Union, Any
import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d

class ConvBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        pre_act_normalize: bool = True,
        normalization: nn.Module = None,
        **kwargs
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = not pre_act_normalize,
            **kwargs
        )
        if normalization:
            self.normalization = deepcopy(normalization)
        else:
            self.normalization = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace = True)
        self.pre_act_normalize = pre_act_normalize
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if self.pre_act_normalize:
            x = self.normalization(x)
            x = self.relu(x)
        else:
            x = self.relu(x)
            x = self.normalization(x)

        return x

if __name__ == '__main__':
    block = ConvBlock2d(16, 32, 3, 1, 1, normalization = nn.GroupNorm(8, 32))
    print(block(torch.randn(8, 16, 64, 64)).shape)