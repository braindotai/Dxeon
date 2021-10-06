from copy import deepcopy
import torch
from torch import nn

class ResidualBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        normalization: nn.Module = None,
        num_blocks: int = 1,
    ):
        super().__init__()

        blocks = []
        for _ in range(num_blocks - 1):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = kernel_size // 2,
                        bias = not isinstance(normalization, nn.BatchNorm2d)
                    ),
                    self._get_normalization(normalization, in_channels),
                    nn.ReLU(True),
                )
            )
        blocks.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = kernel_size // 2,
                    bias = not isinstance(normalization, nn.BatchNorm2d)
                ),
                self._get_normalization(normalization, in_channels),
            )
        )

        self.relu = nn.ReLU(True)
        self.blocks = nn.Sequential(*blocks)
    
    def _get_normalization(self, normalization, in_channels):
        if normalization:
            return deepcopy(normalization)
        return nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(x)
        y += x

        out = self.relu(y)
        
        return out

if __name__ == '__main__':
    from torchinfo import summary
    block = ResidualBlock2d(16, num_blocks = 2, normalization = nn.GroupNorm(8, 16))
    summary(block, (8, 16, 64, 64))
    # print(block(torch.randn(8, 16, 64, 64)).shape)