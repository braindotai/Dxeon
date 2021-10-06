import torch
from torch import nn
from copy import deepcopy

class DenseBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
        num_blocks: int = 1,
        normalization: nn.Module = None,
    ):
        super().__init__()

        convs = []

        for conv_idx in range(num_blocks):
            if normalization:
                convs.append(deepcopy(normalization))
            else:
                convs.append(nn.BatchNorm2d(in_channels))

            convs.append(nn.ReLU(inplace = True))
            
            if conv_idx == num_blocks - 1:
                convs.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = padding,
                        bias = False
                    )
                )
            else:
                convs.append(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = padding,
                        bias = False
                    )
                )
        
        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.convs(x)
        y = self.dropout(y)
        y = torch.cat([x, y], 1)
        return y

if __name__ == '__main__':
    from torchinfo import summary
    block = DenseBlock2d(16, 32, 3, 1, 1, 0.5, num_blocks = 3, normalization = nn.GroupNorm(8, 16))
    summary(block, (8, 16, 64, 64))
    # print(block(torch.randn(8, 16, 64, 64)).shape)