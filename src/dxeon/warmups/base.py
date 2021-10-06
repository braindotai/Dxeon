from typing import Any, Dict
from torch import optim

class Base:
    def __init__(self, optimizer: optim.Optimizer, base_lr: float = 1e-6, **params):
        self.optimizer = optimizer
        self.params = params
        self.last_step = -1
        self.base_lr = base_lr
        self.final_lr = optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self):
        # if self.optimizer.param_groups[0]['lr'] <= self.final_lr:
        self.last_step += 1
        self.optimizer.param_groups[0]['lr'] = self.update_lr(**self.params)

    def update_factor(self, **params):
        raise NotImplementedError