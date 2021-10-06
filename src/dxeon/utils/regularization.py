import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout

def l1_weight_regularization(module: nn.Module, alpha: float = 0.001):
    loss = 0.0
    for parameter in module.parameters():
        if parameter.requires_grad:
            loss += parameter.abs().sum()
    
    return alpha * loss

def l2_weight_regularization(module: nn.Module, alpha: float = 0.001):
    loss = 0.0
    for parameter in module.parameters():
        if parameter.requires_grad:
            loss += parameter.pow(2).sum()
    
    return alpha * loss

def l1l2_weight_regularization(module: nn.Module, alpha: float = 0.001):
    loss = 0.0
    for parameter in module.parameters():
        if parameter.requires_grad:
            loss += 0.5 * (parameter.pow(2).sum() + parameter.abs().sum())
    
    return alpha * loss

def l1_activity_regularization(outputs: torch.Tensor, loss_holder: torch.Tensor, alpha: float = 0.001):
    loss_holder += alpha * outputs.abs().sum()

def l2_activity_regularization(outputs: torch.Tensor, loss_holder: torch.Tensor, alpha: float = 0.001):
    loss_holder += alpha * outputs.pow(2).sum()

def l1l2_activity_regularization(outputs: torch.Tensor, loss_holder: torch.Tensor, alpha: float = 0.001):
    loss_holder += alpha * (outputs.abs().sum() + outputs.pow(2).sum())

class DropConnect(nn.Module):
    '''
    https://discuss.pytorch.org/t/dropconnect-implementation/70921
    '''
    def __init__(self, module: nn.Module, p: float = 0.1, inplace = True):
        super().__init__()

        self.module = module
        self.p = p
        self.inplace = inplace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.module.training:
            with torch.no_grad():
                original_params = []
                for param in self.module.parameters():
                    original_params.append(param.clone())
                    param.copy_(F.dropout(param, p = self.p, inplace = self.inplace) * (1 - self.p))
            
        out = self.module(x)
        
        if self.module.training:
            with torch.no_grad():
                for original_param, param in zip(original_params, self.module.parameters()):  
                    param.copy_(original_param)
        
        return out

if __name__ == '__main__':
    conv = nn.Conv2d(3, 3, 1, 1)
    x = torch.ones(1, 3, 4, 4)
    y = conv(x)
    print(y)
    dropconnect_conv = DropConnect(conv, 0.5)
    print(dropconnect_conv(x))
    print(l1_weight_regularization(conv))
    print(l2_weight_regularization(conv))
    print(l1l2_weight_regularization(conv))

    loss = torch.tensor(0.0)
    l1_activity_regularization(y, loss)
    print(loss)
    l2_activity_regularization(y, loss)
    print(loss)
    l1l2_activity_regularization(y, loss)
    print(loss)