from .. import utils
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy

def compute_guided_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    steps: int = 100,
    has_classes: bool = True,
    class_idx: int = None,
    device: str = 'cuda',
    visualize: bool = True,
) -> torch.Tensor:
    
    model = deepcopy(model)

    def backward_hook(module, input_grads, output_grads):
        if isinstance(module, nn.ReLU):
            return F.relu(input_grads[0]),

    for module in model.modules():
        module.register_backward_hook(backward_hook)

    baseline = torch.zeros_like(input_tensor)
    interpolated_images = torch.stack(
        [
            baseline + alpha * (input_tensor - baseline)
            for alpha in torch.arange(0, 1.0 + 1.0 / steps, 1.0 / steps)
        ],
        dim = 0
    )
    interpolated_images.requires_grad_()

    if next(model.parameters()).device != device:
        outputs = model.to(device)(interpolated_images.to(device))
    else:
        outputs = model(interpolated_images.to(device))

    if has_classes:
        class_idx = class_idx if class_idx else outputs[-1].argmax(0)
        integrated_gradients = torch.autograd.grad(outputs.softmax(1)[:, class_idx].sum(), interpolated_images)[0].mean(dim = 0)
    else:
        integrated_gradients = torch.autograd.grad(outputs.sum(), interpolated_images)[0].mean(dim = 0)

    integrated_gradients = (integrated_gradients * input_tensor).abs().sum(dim = 0)

    if visualize:
        plt.figure(figsize = (7, 7))
	
        plt.subplot(2, 2, 1)
        plt.imshow(utils.image.get_plt_image(input_tensor))
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(utils.image.get_plt_image(integrated_gradients), cmap = 'inferno')
        plt.title('Guided IG')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(utils.image.get_plt_image(input_tensor))
        plt.imshow(utils.image.get_plt_image(integrated_gradients), cmap = 'inferno', alpha = 0.8)
        plt.title('Guided IG Overlay')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        maps = utils.image.normalize(input_tensor) * integrated_gradients
        plt.imshow(utils.image.get_plt_image(maps))
        plt.title('Guided IG Mapped Inputs')
        plt.axis('off')

        plt.show()

    return integrated_gradients.detach()