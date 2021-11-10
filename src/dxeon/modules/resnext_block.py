import torch
from torch import nn
from copy import deepcopy

class ResNextBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        groups: int = 8,
        downsample: bool = False,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(in_channels * 2)
        
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size = 3, stride = 2 if downsample else 1, padding = 1, groups = groups, bias = False)
        self.bn2 = nn.BatchNorm2d(in_channels * 2)
        
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 2, bias = False),
                nn.BatchNorm2d(in_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)

        if self.downsample is not None:
            residual = self.downsample(x)
            y = torch.cat((y, residual), dim = 1)
        else:
            y = y + residual
        
        out = self.relu(y)

        return out

if __name__ == '__main__':
    from torchinfo import summary

    block = ResNextBlock2d(16, groups = 8)
    summary(block, (8, 16, 64, 64))
    block = ResNextBlock2d(16, groups = 8, downsample = True)
    summary(block, (8, 16, 64, 64))
    
    # print(block(torch.randn(8, 16, 64, 64)).shape)