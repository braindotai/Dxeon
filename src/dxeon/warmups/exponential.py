from typing import Any, Dict
import math
from torch import optim
from .base import Base

class Exponential(Base):
    def __init__(self, optimizer: optim.Optimizer, base_lr: float = 1e-6, **params):
        super().__init__(optimizer, base_lr, **params)

    def update_lr(self, max_steps):
        raise NotImplementedError
