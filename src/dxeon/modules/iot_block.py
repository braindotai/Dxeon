import torch
from torch import nn
from copy import deepcopy

class IOTBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_kernel_size: int = 3,
        normalization: nn.Module = None,
        num_blocks: int = 1,
    ):
        super().__init__()

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    self._get_normalization(normalization, in_channels),
                    nn.ReLU(True),
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size = (1, base_kernel_size),
                        stride = 1,
                        padding = (0, base_kernel_size // 2),
                    ),
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size = (base_kernel_size, 1),
                        stride = 1,
                        padding = (base_kernel_size // 2, 0),
                        bias = False
                    )
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(x)
        out = x + y
        return out
    
    def _get_normalization(self, normalization, in_channels):
        if normalization:
            return deepcopy(normalization)
        
        return nn.BatchNorm2d(in_channels)

if __name__ == '__main__':
    from torchinfo import summary
    block = IOTBlock2d(16, normalization = nn.GroupNorm(8, 16), num_blocks = 5)
    summary(block, (8, 16, 64, 64))
    # print(block(torch.randn(8, 16, 64, 64)).shape)