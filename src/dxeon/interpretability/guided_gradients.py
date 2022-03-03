import os

from cv2 import grabCut
from numpy import gradient
from .. import utils
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms

def prep_grads(grads):
    invTrans = transforms.Compose([
        transforms.Normalize(mean = [0., 0., 0.], std = [1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean = [-0.485, -0.456, -0.406], std = [1., 1., 1.])
    ])
    out = invTrans(grads.unsqueeze(0))[0]
    out = out.detach()
    return out

def norm_flat_image(grads):
    grads_norm = prep_grads(grads)
    grads_norm = grads_norm[0, :, :, ] + grads_norm[1, :, :, ] + grads_norm[2, :, :, ]
    grads_norm = (grads_norm - grads_norm.min()) / (grads_norm.max()- grads_norm.min())
    return grads_norm

def compute_guided_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    has_classes: bool = True,
    class_idx: int = None,
    device: str = 'cuda',
    visualize: bool = True,
    save_path: os.PathLike = None,
) -> torch.Tensor:

    model.zero_grad()

    def backward_hook(module, input_grads, output_grads):
        if isinstance(module, nn.ReLU):
            return F.relu(input_grads[0]),

    handels = []
    for module in model.modules():
        handels.append(module.register_backward_hook(backward_hook))

    input_tensor.requires_grad_()

    if next(model.parameters()).device != device:
        outputs = model.to(device)(input_tensor.unsqueeze(0).to(device))
    else:
        outputs = model(input_tensor.unsqueeze(0).to(device))

    if has_classes:
        class_idx = class_idx if class_idx else outputs[-1].argmax(0)
        print(class_idx)
        guided_gradients = torch.autograd.grad(outputs[0, class_idx].sum(), input_tensor)[0]
    else:
        guided_gradients = torch.autograd.grad(outputs.sum(), input_tensor)[0]
    
    input_tensor.requires_grad = False

    for handle in handels:
        handle.remove()

    if visualize:
        input_tensor = input_tensor.detach()
        
        plt.figure(figsize = (7, 7))
	
        plt.subplot(2, 2, 1)
        plt.imshow(utils.image.get_plt_image(input_tensor))
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(utils.image.get_plt_image(guided_gradients), vmin = 0.3, vmax = 0.7, cmap = 'gray')
        plt.title('Guided Gradients')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(utils.image.get_plt_image(input_tensor))
        plt.imshow(utils.image.get_plt_image(guided_gradients), cmap = 'viridis', alpha = 0.8)
        plt.title('Guided Gradients Overlay')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        maps = utils.image.normalize(input_tensor) * guided_gradients
        plt.imshow(utils.image.get_plt_image(maps))
        plt.title('Guided Gradients Mapped inputs')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path)

        plt.show()

    return guided_gradients.detach()