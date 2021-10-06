# from ..visualize.image import normalize_image
from torchvision import models
from .. import utils
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compute_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    steps: int = 100,
    has_classes: bool = True,
    class_idx: int = None,
    device: str = 'cuda',
    visualize: bool = True,
) -> torch.Tensor:
    baseline = torch.zeros_like(input_tensor)

    def conv_backward_hook(module, input_grads, output_grads):
        input_grads = input_grads[0]
        # max_val = input_grads.max()
        # print(input_grads.max(), input_grads.min())
        # input_grads = (input_grads - input_grads.mean())/(input_grads.std())
        # input_grads = (input_grads - input_grads.min())/(input_grads.max() - input_grads.min()) # 0, 1
        # print(input_grads.max(), input_grads.min())
        # input_grads = input_grads * (2.0) - 1.0 # -1, 1
        # print(input_grads.max(), input_grads.min(), '....')
        # input_grads = F.relu(input_grads)

        return input_grads,
    
    def relu_backward_hook(module, input_grads, output_grads):
        input_grads = input_grads[0]
        input_grads = F.relu(input_grads)

        return input_grads,

    for module in model.modules():
        # if isinstance(module, nn.Conv2d):
        #     module.register_full_backward_hook(conv_backward_hook)
        if isinstance(module, nn.ReLU):
            module.register_backward_hook(relu_backward_hook)

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
    # integrated_gradients = (integrated_gradients).sum(dim = 0)

    if visualize:
        plt.figure(figsize = (7, 7))
	
        plt.subplot(2, 2, 1)
        plt.imshow(utils.image.get_plt_image(input_tensor))
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(utils.image.get_plt_image(integrated_gradients), cmap = 'inferno')
        plt.title('Integrated Gradients')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(utils.image.get_plt_image(input_tensor))
        plt.imshow(utils.image.get_plt_image(integrated_gradients), cmap = 'inferno', alpha = 0.8)
        plt.title('Integrated Gradients Overlay')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        maps = utils.image.normalize(input_tensor) * integrated_gradients
        plt.imshow(utils.image.get_plt_image(maps))
        plt.title('Integrated Gradients Mapped Inputs')
        plt.axis('off')

        plt.show()

    return integrated_gradients.detach()