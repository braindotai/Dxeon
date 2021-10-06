import torch
from torch import nn
import torch.nn.functional as F

class WeightStandardisationConv2d(nn.Conv2d):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        weight_mean = weight.mean(dim = 1, keepdim = True)\
                            .mean(dim = 2, keepdim = True)\
                            .mean(dim = 3, keepdim = True)

        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim = 1).view(-1, 1, 1, 1) + 1e-6
        weight = weight / std.expand_as(weight)
        
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

if __name__ == '__main__':
    from torchinfo import summary

    block = WeightStandardisationConv2d(64, 128, 3, 1, 1)
    summary(block, (8, 64, 64, 64))