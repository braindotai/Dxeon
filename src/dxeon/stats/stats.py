from typing import List, Tuple, Union
import torch
from torch import nn
from torchinfo import summary
from ..utils.model import get_device

def summarize(item: Union[nn.Module, torch.Tensor], input_size: Union[List, Tuple] = None, device: str = 'cpu', name = '', **kwargs):
    if isinstance(item, nn.Module):
        _original_device = get_device(item)
        summary(item, input_size = input_size, device = device, verbose = 1, **kwargs)

        if device != _original_device:
            item.to(_original_device)
        
    elif isinstance(item, torch.Tensor):
        print(f'{name} Shape\t:', item.shape)
        print(f'{name} Dtype\t:', item.dtype)
        print(f'{name} Device\t:', item.device)
        print(f'{name} Numel\t:', item.numel())
        print(f'{name} Max\t:', item.max().item())
        print(f'{name} Min\t:', item.min().item())
        print(f'{name} Sum\t:', item.sum().item())
        print(f'{name} Mean\t:', item.float().mean().item())
        print(f'{name} Std\t:', item.float().std().item())
        print(f'{name} Median\t:', item.float().median().item(), '\n')