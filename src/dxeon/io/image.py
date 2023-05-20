
import os
import requests
from io import BytesIO

import torch
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Union

from ..utils import resize_cv2, resize_pil, pil_to_torch


def read_pil(image_path: str, size: Union[int, Tuple[int, int], List[int]] = None, interpolation: str = 'bilinear') -> Image.Image:
    assert os.path.isfile(image_path), f'\n\nImage with path "{image_path}" does not exist.\n'
    image = Image.open(image_path).convert('RGB')
    if size:
        image = resize_pil(image, size, interpolation)
    return image
    
def read_cv2(image_path: str, size: Union[int, Tuple[int, int], List[int]] = None, interpolation: str = 'linear') -> np.ndarray:
    assert os.path.isfile(image_path), f'\n\nImage with path "{image_path}" does not exist.\n'
    image = cv2.imread(image_path)
    if size:
        image = resize_cv2(image, size, interpolation)
    return image

def read_torch(image_path: str, size: Union[int, Tuple[int, int], List[int]] = None, interpolation: str = 'linear') -> np.ndarray:
    assert os.path.isfile(image_path), f'\n\nImage with path "{image_path}" does not exist.\n'
    pil_image = read_pil(image_path, size, interpolation)
    return pil_to_torch(pil_image)

def from_url_to_pil(url: str, size: Union[int, Tuple[int, int], List[int]] = None, interpolation: str = 'bilinear') -> Image.Image:
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    if size:
        image = resize_pil(image, size, interpolation)
    return image

def from_url_to_cv2(url: str, size: Union[int, Tuple[int, int], List[int]] = None, interpolation: str = 'linear') -> np.ndarray:
    response = requests.get(url)
    image = cv2.imdecode(np.frombuffer(response.content, dtype = np.uint8), -1)
    if size:
        image = resize_cv2(image, size, interpolation)
    return image

def from_url_to_torch(url: str, size: Union[int, Tuple[int, int], List[int]] = None, interpolation: str = 'bilinear') -> np.ndarray:
    pil_image = from_url_to_pil(url, size, interpolation)
    return pil_to_torch(pil_image)

def write(image: Union[Image.Image, np.ndarray, torch.Tensor], image_path: str) -> None:
    if isinstance(image, Image.Image):
        image.save(image_path)
    elif isinstance(image, np.ndarray) or type(image).__module__ == np.__name__:
        image = image.astype('float32')
        image = (image - image.min())/image.ptp()
        image *= 255.0
        image = image.astype('uint8')
        cv2.imwrite(image_path, image)
    else:
        image = image.permute(1, 2, 0).numpy().astype('float32')
        image = (image - image.min())/image.ptp()
        image *= 255.0
        image = image.astype('uint8')
        cv2.imwrite(image_path, image)
