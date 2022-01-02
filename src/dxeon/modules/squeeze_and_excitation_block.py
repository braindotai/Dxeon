import torch
from torch import nn

class ChannelSqueezeAndExcitationBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: float = 2.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(in_channels, int(in_channels // reduction_ratio), bias=True)
        self.fc2 = nn.Linear(int(in_channels // reduction_ratio), in_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, _, _ = x.size()
        squeeze_tensor = x.view(batch_size, in_channels, -1).mean(dim=2)

        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        out = x * fc_out_2.view(batch_size, in_channels, 1, 1)

        return out

class SpatialSqueezeAndExcitationBlock2d(nn.Module):
    def __init__(self, reduction_channels: int = 32):
        super().__init__()

        self.conv1 = nn.Conv2d(1, reduction_channels, 3, 2, 1)
        self.conv2 = nn.ConvTranspose2d(reduction_channels, 1, 4, 2, 1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_tensor = x.sum(dim = 1) / x.size(1)
        conv_out1 = self.relu(self.conv1(squeeze_tensor.unsqueeze(1)))
        conv_out2 = self.sigmoid(self.conv2(conv_out1))

        out = x * conv_out2

        return out


if __name__ == '__main__':
    from torchinfo import summary
    block = ChannelSqueezeAndExcitationBlock2d(256, 4)
    summary(block, (8, 256, 64, 64))
    block = SpatialSqueezeAndExcitationBlock2d()
    summary(block, (8, 256, 64, 64))