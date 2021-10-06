from typing import Any, Dict
from torch import optim
from .base import Base

class Linear(Base):
    def __init__(self, optimizer: optim.Optimizer, base_lr: float = 1e-6, **params):
        super().__init__(optimizer, base_lr, **params)

    def update_lr(self, max_steps):
        return self.base_lr + ((self.final_lr - self.base_lr) / max_steps) * self.last_step