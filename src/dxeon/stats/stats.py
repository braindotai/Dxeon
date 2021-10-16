from typing import List, Tuple, Union
import torch
from torch import nn
from torchinfo import summary

def summarize(item: Union[nn.Module, torch.Tensor], input_size: Union[List, Tuple] = None, device: str = 'cpu', name = ''):
    if isinstance(item, nn.Module):
        summary(item, input_size = input_size, device = device)
    
    elif isinstance(item, torch.Tensor):
        print(f'{name} Shape\t:', item.shape)
        print(f'{name} Dtype\t:', item.dtype)
        print(f'{name} Device\t:', item.device)
        print(f'{name} Numel\t:', item.numel())
        print(f'{name} Max\t:', item.max().item())
        print(f'{name} Min\t:', item.min().item())
        print(f'{name} Sum\t:', item.sum().item())
        print(f'{name} Mean\t:', item.mean().item())
        print(f'{name} Std\t:', item.std().item())
        print(f'{name} Median\t:', item.median().item())