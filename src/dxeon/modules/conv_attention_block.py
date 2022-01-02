import torch
from torch import nn

class AttentionConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction_channels: int,
    ):
        super().__init__()

        self.query_conv = nn.Conv2d(in_channels, reduction_channels, 1, 1, 0, bias = True)
        self.key_conv = nn.Conv2d(in_channels, reduction_channels, 1, 1, 0, bias = True)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias = True)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.reduction_channels = reduction_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        bc = size[:2]
        q = self.query_conv(x).view(bc[0], self.reduction_channels, -1)
        k = self.key_conv(x).view(bc[0], self.reduction_channels, -1)
        v = self.value_conv(x).view(*bc, -1)

        correlations = torch.bmm(q.permute(0, 2, 1), k)
        beta = torch.softmax(correlations, dim = 1) # (-1, reduction_channels, h * w)
        attention = self.gamma * torch.bmm(v, beta) # (-1, in_channels, h * w)

        out = (attention.view(*size) + x).contiguous()

        return out

if __name__ == '__main__':
    from torchinfo import summary
    block = AttentionConv2d(256, 32)
    summary(block, (8, 256, 64, 64))