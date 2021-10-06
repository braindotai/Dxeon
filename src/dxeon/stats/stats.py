from typing import List, Tuple, Union
import torch
from torch import nn
from torchinfo import summary

def summarize(item: Union[nn.Module, torch.Tensor], input_size: Union[List, Tuple] = None, device: str = 'cpu'):
    if isinstance(item, nn.Module):
        summary(item, input_size = input_size, device = device)
    else:
        print('Shape\t:', item.shape)
        print('Dtype\t:', item.dtype)
        print('Device\t:', item.device)
        print('Numel\t:', item.numel())
        print('Max\t:', item.max().item())
        print('Min\t:', item.min().item())
        print('Sum\t:', item.sum().item())
        print('Mean\t:', item.mean().item())
        print('Std\t:', item.std().item())
        print('Median\t:', item.median().item())