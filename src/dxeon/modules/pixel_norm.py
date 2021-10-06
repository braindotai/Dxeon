import torch
from torch import nn

class PixelWiseNorm2d(nn.Module):
    def __init__(self):
        super(PixelWiseNorm2d, self).__init__()
        
    def forward(self, x):
        c = torch.mean(x**2, dim = 1, keepdim = True)
        return x / (c + 1e-8) ** 0.5