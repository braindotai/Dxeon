import torch
from torch import nn
from torch.nn.modules.instancenorm import InstanceNorm1d

def _get_basic_block(in_channels, out_channels, kernel_size, stride = 1, padding = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

class InceptionBlockA(nn.Module):
    out_channel_factor = 1.0

    def __init__(self, in_channels):
        super().__init__()
        
        self.branch0 = _get_basic_block(in_channels, in_channels // 4, kernel_size = 1, stride = 1)

        self.branch1 = nn.Sequential(
            _get_basic_block(in_channels, in_channels // 6, kernel_size = 1, stride = 1),
            _get_basic_block(in_channels // 6, in_channels // 4, kernel_size = 3, stride = 1, padding = 1)
        )

        self.branch2 = nn.Sequential(
            _get_basic_block(in_channels, in_channels // 6, kernel_size = 1, stride = 1),
            _get_basic_block(in_channels // 6, in_channels // 4, kernel_size = 3, stride = 1, padding = 1),
            _get_basic_block(in_channels // 4, in_channels // 4, kernel_size = 3, stride = 1, padding = 1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride = 1, padding = 1, count_include_pad = False),
            _get_basic_block(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        )        

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        
        return out


class InceptionBlockB(nn.Module):
    out_channel_factor = 1.125

    def __init__(self, in_channels):
        super().__init__()

        self.branch0 = _get_basic_block(in_channels, int(in_channels / 2.666), kernel_size = 1, stride = 1)

        self.branch1 = nn.Sequential(
            _get_basic_block(in_channels, int(in_channels / 5.333), kernel_size = 1, stride = 1),
            _get_basic_block(int(in_channels / 5.333), int(in_channels / 4.571), kernel_size = (1, 7), stride = 1, padding = (0, 3)),
            _get_basic_block(int(in_channels / 4.571), in_channels // 4, kernel_size = (7, 1), stride = 1, padding = (3, 0))
        )

        self.branch2 = nn.Sequential(
            _get_basic_block(in_channels, int(in_channels / 5.333), kernel_size = 1, stride = 1),
            _get_basic_block(int(in_channels / 5.333), int(in_channels / 5.333), kernel_size = (7, 1), stride = 1, padding = (3, 0)),
            _get_basic_block(int(in_channels / 5.333), int(in_channels / 4.571), kernel_size = (1, 7), stride = 1, padding = (0, 3)),
            _get_basic_block(int(in_channels / 4.571), int(in_channels / 4.571), kernel_size = (7, 1), stride = 1, padding = (3, 0)),
            _get_basic_block(int(in_channels / 4.571), in_channels // 4, kernel_size = (1, 7), stride = 1, padding = (0, 3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride = 1, padding = 1, count_include_pad=False),
            _get_basic_block(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)

        return out

class InceptionBlockC(nn.Module):
    out_channel_factor = 1.125

    def __init__(self, in_channels):
        super().__init__()

        self.branch0 = _get_basic_block(in_channels, in_channels // 6, kernel_size = 1, stride = 1)

        self.branch1_0 = _get_basic_block(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.branch1_1a = _get_basic_block(in_channels // 4, in_channels // 6, kernel_size = (1, 3), stride = 1, padding = (0,1))
        self.branch1_1b = _get_basic_block(in_channels // 4, in_channels // 6, kernel_size = (3, 1), stride = 1, padding = (1,0))

        self.branch2_0 = _get_basic_block(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.branch2_1 = _get_basic_block(in_channels // 4, int(in_channels / 3.4285), kernel_size = (3, 1), stride = 1, padding = (1,0))
        self.branch2_2 = _get_basic_block(int(in_channels / 3.4285), in_channels // 3, kernel_size = (1, 3), stride = 1, padding = (0,1))
        self.branch2_3a = _get_basic_block(in_channels // 3, in_channels // 4, kernel_size = (1, 3), stride = 1, padding = (0,1))
        self.branch2_3b = _get_basic_block(in_channels // 3, in_channels // 4, kernel_size = (3, 1), stride = 1, padding = (1,0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride = 1, padding = 1, count_include_pad = False),
            _get_basic_block(in_channels, in_channels // 6, kernel_size = 1, stride = 1)
        )

        self.register_buffer('out_channels', torch.tensor((in_channels // 2) + (in_channels // 3) + (in_channels // 2)))

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)

        return out
    
class InceptionReductionBlockA(nn.Module):
    out_channel_factor = 3

    def __init__(self, in_channels):
        super().__init__()
        self.branch0 = _get_basic_block(in_channels, in_channels, kernel_size = 3, stride = 2) # half

        self.branch1 = nn.Sequential(
            _get_basic_block(in_channels, in_channels, kernel_size = 1, stride = 1, padding = 0), # same
            _get_basic_block(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1), # same
            _get_basic_block(in_channels, in_channels, kernel_size = 3, stride = 2, padding = 1) # half
        )

        self.branch2 = nn.MaxPool2d(2, stride = 2) # half

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)

        return out

class InceptionReductionBlockB(nn.Module):
    out_channel_factor = 1.5

    def __init__(self, in_channels):
        super().__init__()

        self.branch0 = nn.Sequential(
            _get_basic_block(in_channels, int(in_channels / 5.333), kernel_size = 1, stride = 1),
            _get_basic_block(int(in_channels / 5.333), int(in_channels / 5.333), kernel_size = 3, stride = 2)
        )

        self.branch1 = nn.Sequential(
            _get_basic_block(in_channels, in_channels // 4, kernel_size = 1, stride = 1),
            _get_basic_block(in_channels // 4, in_channels // 4, kernel_size = (1, 7), stride = 1, padding = (0, 3)),
            _get_basic_block(in_channels // 4, int(in_channels / 3.2), kernel_size = (7, 1), stride = 1, padding = (3, 0)),
            _get_basic_block(int(in_channels / 3.2), int(in_channels / 3.2), kernel_size = 3, stride = 2)
        )

        self.branch2 = nn.MaxPool2d(2, stride = 2, padding = 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)

        return out


if __name__ == '__main__':
    block1 = InceptionBlockA(64)
    block2 = InceptionBlockB(64)
    block3 = InceptionBlockC(64)
    
    block4 = InceptionReductionBlockA(64)
    block5 = InceptionReductionBlockB(64)

    print('InceptionBlockA:', block1(torch.ones(8, 64, 16, 16)).shape)
    print('InceptionBlockB:', block2(torch.ones(8, 64, 16, 16)).shape)
    print('InceptionBlockC:', block3(torch.ones(8, 64, 16, 16)).shape)
    print('InceptionReductionBlockA:', block4(torch.ones(8, 64, 16, 16)).shape)
    print('InceptionReductionBlockB:', block5(torch.ones(8, 64, 16, 16)).shape)