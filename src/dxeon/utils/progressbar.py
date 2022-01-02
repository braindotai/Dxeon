from typing import Iterable
from tqdm.auto import tqdm

def progressbar(iterator: Iterable, epoch: int = None, total_epochs: int = None, desc: str = None, ncols: int = 1000):
    return tqdm(
        iterator,
        desc = desc if desc else f'Epoch({epoch + 1}/{total_epochs})',
        ncols = ncols
    )