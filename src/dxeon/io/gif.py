import os
import torch
import imageio
from typing import List, Union
from .. import utils
import numpy as np
from PIL import Image, ImageSequence

def write(images: List[Union[np.ndarray, torch.Tensor]], output_path: str, fps: int = 10):
    images_ = []
    for img in images:
        if isinstance(img, torch.Tensor):
            if len(img.shape) == 3:
                img = img.permute(1, 2, 0).numpy()
            elif img.shape[0] == 1:
                img = img[0].numpy()
            else:
                img = img.numpy()
        elif img.shape[0] == 1:
            img = img[0]
        images_.append((utils.normalize(img) * 255.0).astype('uint8'))
    imageio.mimsave(output_path, images_, fps = fps)

def read(path: os.PathLike, transform = None):
    images_ = []
    for frame in ImageSequence.Iterator(Image.open(path)):
        if transform:
            frame = transform(frame)
        images_.append(frame)
    return images_