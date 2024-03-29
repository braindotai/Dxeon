import os
from .. import utils
import torch
from torch import nn
import matplotlib.pyplot as plt

def compute_gradient_times_input(
    model: nn.Module,
    input_tensor: torch.Tensor,
    has_classes: bool = True,
    class_idx: int = None,
    device: str = 'cuda',
    visualize: bool = True,
    save_path: os.PathLike = None,
) -> torch.Tensor:
    '''
    https://arxiv.org/pdf/1312.6034.pdf
    '''

    model.zero_grad()

    input_tensor.requires_grad_()

    if next(model.parameters()).device != device:
        outputs = model.to(device)(input_tensor.unsqueeze(0).to(device))
    else:
        outputs = model(input_tensor.unsqueeze(0).to(device))

    if has_classes:
        class_idx = class_idx if class_idx is not None else outputs[-1].argmax(0)
        saliency_maps = torch.autograd.grad(outputs.softmax(1)[:, class_idx].sum(), input_tensor)[0]
    else:
        saliency_maps = torch.autograd.grad(outputs.sum(), input_tensor)[0]
    
    input_tensor.requires_grad = False
    
    saliency_maps *= input_tensor

    saliency_maps = (saliency_maps - saliency_maps.min()) / (saliency_maps.max() - saliency_maps.min())
    
    if visualize:
        input_tensor = input_tensor.detach()
        
        plt.figure(figsize = (7, 7))
	
        plt.subplot(2, 2, 1)
        plt.imshow(utils.image.get_plt_image(input_tensor))
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(utils.image.get_plt_image(saliency_maps), cmap = 'inferno')
        plt.title('Saliencey Maps')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(utils.image.get_plt_image(input_tensor))
        plt.imshow(utils.image.get_plt_image(saliency_maps), cmap = 'inferno', alpha = 0.8)
        plt.title('Saliencey Maps Overlay')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        maps = utils.image.normalize(input_tensor) * saliency_maps
        plt.imshow(utils.image.get_plt_image(maps))
        plt.title('Saliencey Mapped inputs')
        plt.axis('off')

        plt.savefig('old.jpg')

        if save_path:
            plt.savefig(save_path)

        plt.show()

    return saliency_maps.detach()