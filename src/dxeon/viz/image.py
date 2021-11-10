from typing import Dict, List, Union, Tuple
import cv2
import torch
import numpy as np
from PIL import Image
from .. import utils
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    title: str = None,
    size: Tuple[int] = None,
    bgr2rgb = False,
    normalize: bool = True,
    save: str = None,
    cmap = None,
    alpha = 1.0,
    show = True,
):
    if size:
        plt.figure(figsize = size)
    image = utils.image.get_plt_image(image, bgr2rgb, normalize)
    cmap = cmap or ('gray' if len(image.shape) == 2 else None)
    plt.imshow(image, cmap = cmap, alpha = alpha)
    plt.axis('off')
    
    if title is not None:
        plt.title(title)
    
    if save:
        plt.savefig(save)
    
    if show:
        plt.show()

def _get_cmap_alpha(cmap, alpha, cols):
    if not isinstance(cmap, str):
        cmap = cmap or [None] * cols
    else:
        cmap = [cmap] * cols
    if not isinstance(alpha, float):
        alpha = alpha or [None] * cols
    else:
        alpha = [alpha] * cols

    return cmap, alpha

def image_stack(
    pairs_lists: List[Union[Image.Image, np.ndarray, torch.Tensor]],
    titles: str = None,
    size: Tuple[int] = (8, 8),
    bgr2rgb: bool = True,
    normalize: bool = True,
    cmap = None,
    alpha = 1.0,
) -> None:
    assert len(pairs_lists[0]) == len(pairs_lists[1]), f'\n\nPairs mismatch found! Make sure no. of image in both lists are same.\n'

    if titles:
        assert len(pairs_lists) == len(titles), f'\n\nTitles must be provided for each image columns, found total "{len(titles)}" titles, while total image columns are "{len(pairs_lists)}".\n'
    
    cols = len(pairs_lists)
    
    cmap, alpha = _get_cmap_alpha(cmap, alpha, cols)

    for images in zip(*pairs_lists):
        plt.figure(figsize = size)
        for idx, image in enumerate(images):
            plt.subplot(1, cols, idx + 1)
            image = utils.image.get_plt_image(image, bgr2rgb, normalize)
            plt.imshow(image, cmap = cmap[idx], alpha = alpha[idx])
            plt.axis('off')
            if titles:
                plt.title(titles[idx])
        plt.show()

def image_grid(
    grid_list: List[Union[Image.Image, np.ndarray, torch.Tensor]],
    shape: Tuple[int] = (6, 6),
    size: int = 10,
    title: str = None,
    bgr2rgb: bool = True,
    normalize: bool = True,
    save: bool = None,
    cmap = None,
    alpha = 1.0,
) -> None:
    fig = plt.figure(figsize = (size, size))
    grid = ImageGrid(fig, 111, nrows_ncols = shape, axes_pad = 0.02)

    cmap, alpha = _get_cmap_alpha(cmap, alpha, len(grid_list))

    for idx, (ax, image) in enumerate(zip(grid, grid_list)):
        ax.imshow(
            utils.image.get_plt_image(image, bgr2rgb, normalize),
            cmap = cmap[idx],
            alpha = alpha[idx]
        )
        ax.axis('off')

    if title is not None:
        print(title)
    if save:
        plt.savefig(save)
    plt.show()

def labeled_images(
    image_batch: torch.Tensor,
    label_batch: torch.Tensor,
    shape: Tuple[int],
    label_dict: Dict[int, str] = None,
    size: int = 10,
    title: str = None,
    bgr2rgb: bool = True,
    normalize: bool = True,
    save: bool = None,
    cmap = None,
    alpha = 1.0,
) -> None:

    fig = plt.figure(figsize = (size, size))
    grid = ImageGrid(fig, 111, nrows_ncols = shape, axes_pad = (0.05, 0.5))

    cmap, alpha = _get_cmap_alpha(cmap, alpha, int(shape[0] * shape[1]))

    for idx, (ax, image, label) in enumerate(zip(grid, image_batch, label_batch)):
        ax.imshow(
            utils.image.get_plt_image(image, bgr2rgb, normalize),
            cmap = cmap[idx],
            alpha = alpha[idx]
        )
        ax.axis('off')
        ax.set_title(f'{label}' if label_dict is None else f'{label_dict[label]}')

    if title is not None:
        print(title)
    if save:
        plt.savefig(save)
    plt.show()