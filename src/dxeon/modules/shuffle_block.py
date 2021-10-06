from copy import deepcopy
import torch
from torch import nn

class ShuffleBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        groups: int,
        downsample: bool = False,
        normalization: nn.Module = None,
    ):
        super().__init__()

        self.conv_1x1_block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 1, groups = groups, stride = 1, bias = not isinstance(normalization, nn.BatchNorm2d)),
            self._get_normalization(normalization, in_channels),
            nn.ReLU(True)
        )

        self.conv_3x3_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, groups = in_channels, stride = 2 if downsample else 1, padding = 1, bias = not isinstance(normalization, nn.BatchNorm2d)),
            self._get_normalization(normalization, in_channels),
        )
        self.groups = groups

        self.conv_1x1_block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 1, groups = groups, stride = 1, bias = not isinstance(normalization, nn.BatchNorm2d)),
            self._get_normalization(normalization, in_channels),
        )

        self.downsampler = nn.AvgPool2d(2, 2) if downsample else nn.Identity()
        self.skip_connection = lambda a, b: torch.cat((a, b), dim = 1) if downsample else lambda a, b: a + b
    
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x
    
    def _get_normalization(self, normalization, in_channels):
        if normalization:
            return deepcopy(normalization)
        return nn.BatchNorm2d(in_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_1x1_block1(x)
        y = self.channel_shuffle(y)
        y = self.conv_3x3_block(y)
        y = self.conv_1x1_block2(y)
        
        x = self.downsampler(x)

        out = self.skip_connection(x, y)
        
        return out


if __name__ == '__main__':
    from torchinfo import summary
    block = ShuffleBlock2d(16, 4, True)
    summary(block, (8, 16, 64, 64))