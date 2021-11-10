from copy import deepcopy
import torch
from torch import nn

class DepthwiseSeperableConv2d(nn.Module):
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
        super(DepthwiseSeperableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            groups = in_channels,
            bias = False,
            **kwargs
        )
        self.pointwise = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            bias = False
        )

        self.pre_act_normalize = pre_act_normalize
        
        if normalization:
            self.dw_normalization = deepcopy(normalization)
        else:
            self.dw_normalization = nn.BatchNorm2d(in_channels)
        
        if normalization:
            self.pw_normalization = deepcopy(normalization)
        else:
            self.pw_normalization = nn.BatchNorm2d(out_channels)
            
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.depthwise(x)
        if self.pre_act_normalize:
            x = self.dw_normalization(x)
            x = self.relu(x)
        else:
            x = self.relu(x)
            x = self.dw_normalization(x)

        x = self.pointwise(x)
        if self.pre_act_normalize:
            x = self.pw_normalization(x)
            x = self.relu(x)
        else:
            x = self.relu(x)
            x = self.pw_normalization(x)

        return x

if __name__ == '__main__':
    block = DepthwiseSeperableConv2d(16, 32, 3, 1, 1, 0.5, nn.GroupNorm(8, 32))
    print(block(torch.randn(8, 16, 64, 64)).shape)