import torch
import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, List

def resize_pil(image: Image, size: Union[int, Tuple[int, int], List[int]], interpolation: str = 'bicubuc') -> Image:
    assert type(size) in [tuple, int, list], f'\n\n`size` must be an int or tuple of ints, but got "{type(size)}".\n'

    if interpolation == 'bicubic':
        interpolation_method = Image.BICUBIC
    elif interpolation == 'bilinear':
        interpolation_method = Image.BILINEAR
    elif interpolation == 'antialias':
        interpolation_method = Image.ANTIALIAS 
    elif interpolation == 'nearest':
        interpolation_method = Image.NEAREST
    elif interpolation == 'box':
        interpolation_method = Image.BOX
    elif interpolation == 'hamming':
        interpolation_method = Image.HAMMING
    elif interpolation == 'lanczos':
        interpolation_method = Image.LANCZOS
    else:
        raise ValueError(f'\n\n`interpolation` must be one of bicubic, bilinear, antialias, nearest, box, hamming, or lanczos, but got {interpolation}.\n')
    
    w, h = image.size

    if isinstance(size, int):
        if w > h:
            aspect_ratio = h / w
            dim = (size, int(aspect_ratio * size))
        else:
            aspect_ratio = w / h
            dim = (int(aspect_ratio * size), size)
        image = image.resize(dim, resample = interpolation_method)
    else:
        image = image.resize(size, resample = interpolation_method)

    return image

def resize_cv2(image: Image, size: Union[int, Tuple[int, int], List[int]], interpolation: str = 'linear') -> Image:
    assert type(size) in [tuple, int, list], f'\n\n`size` must be an int or tuple of ints, but got "{type(size)}".\n'
        
    h, w = image.shape[:2]

    if interpolation == 'cubic':
        interpolation_method = cv2.INTER_CUBIC
    elif interpolation == 'linear':
        interpolation_method = cv2.INTER_LINEAR
    elif interpolation == 'area':
        interpolation_method = cv2.INTER_AREA
    else:
        raise ValueError(f'\n\n`interpolation` must be one of cubic, linear or area, but got {interpolation}.\n')

    if isinstance(size, int):
        if w > h:
            aspect_ratio = h / w
            dim = (size, int(aspect_ratio * size))
        else:
            aspect_ratio = w / h
            dim = (int(aspect_ratio * size), size)
        
        image = cv2.resize(image, dim, interpolation = interpolation_method)
    else:
        image = cv2.resize(image, tuple(size), interpolation = interpolation_method)
    return image

def bgr2rgb_cv2(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def rgb2bgr_cv2(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def rgb2gray_cv2(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def bgr2gray_cv2(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def hflip_cv2(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)

def vflip_cv2(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 0)

def rectangle_cv2(image: np.ndarray, pt1, pt2, color = [255, 0, 0], thickness = 3) -> np.ndarray:
    return cv2.rectangle(image, pt1, pt2, color, thickness)

def normalize(image: Union[np.ndarray, torch.Tensor]):
    return (image - image.min()) / (image.max() - image.min())

def pil_to_numpy(pil_image: Image) -> torch.Tensor:
    np_image = np.array(pil_image).astype('float32')
    np_image = (np_image - np_image.min())/(np_image.ptp())
    
    return np_image

def pil_to_torch(pil_image: Image) -> torch.Tensor:
    np_image = pil_to_numpy(pil_image)

    torch_image = torch.tensor(np_image).permute(2, 0, 1)
    
    return torch_image

def torch_to_numpy(torch_image: np.ndarray) -> torch.Tensor:
    torch_image = (torch_image - torch_image.min())/(torch_image.max() - torch_image.min())
    
    if len(torch_image.shape) == 3:
        np_image = torch_image.cpu().permute(1, 2, 0).numpy()
    else:
        np_image = torch_image.numpy()
    
    return np_image
    
def get_plt_image(image, bgr2rgb: bool = False, normalize_image: bool = True):
    if isinstance(image, np.ndarray) and bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if normalize_image and not isinstance(image, Image.Image):
        image = normalize(image)
    
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1, 2, 0).numpy() if len(image.shape) == 3 else image.cpu().numpy()
    
    if isinstance(image, np.ndarray):
        if image.shape[-1] == 1:
            image = image[:, :, 0]
        elif image.shape[0] == 1:
            image = image[0]
    
    return image