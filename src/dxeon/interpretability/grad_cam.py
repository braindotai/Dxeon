import os
import torch
from torch import nn
from .. import utils
import matplotlib.pyplot as plt
import numpy as np
import cv2

def compute_grad_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    has_classes: bool = True,
    class_idx: int = None,
    device: str = 'cuda',
    visualize: bool = True,
    save_path: os.PathLike = None,
) -> torch.Tensor:
    
    model.zero_grad()

    def backward_hook(module, inputs_grad, outputs_grad):
        model._hook_gradients = outputs_grad[0]
    
    def forward_hook(module, inputs, outputs):
        model._hook_activations = outputs
        
    handle1 = getattr(model, layer_name).register_forward_hook(forward_hook)
    handle2 = getattr(model, layer_name).register_backward_hook(backward_hook)

    if next(model.parameters()).device != device:
        outputs = model.to(device)(input_tensor.unsqueeze(0).to(device))
    else:
        outputs = model(input_tensor.unsqueeze(0).to(device))

    if has_classes:
        class_idx = class_idx if class_idx is not None else outputs[0].argmax(0)
        outputs[:, class_idx].backward()
    else:
        outputs[0].sum().backward()

    with torch.no_grad():
        gradients = model._hook_gradients
        pooled_gradients = torch.mean(gradients, dim = [0, 2, 3])
        
        activations = model._hook_activations.detach()
        activations *= pooled_gradients.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        
        activation_maps = torch.sum(activations, dim = 1).squeeze().cpu()
        activation_maps = np.maximum(activation_maps, 0)
        activation_maps /= torch.max(activation_maps)

        heatmap = (255 * activation_maps.squeeze()).type(torch.uint8).cpu().numpy()
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
        b, g, r = heatmap.split(1)
        heatmap = torch.cat([r, g, b]) * 1.0

        heatmap = utils.image.resize_cv2(heatmap.permute(1, 2, 0).numpy(), list(input_tensor.shape[1:]), 'cubic')
        activation_maps = utils.image.resize_cv2(activation_maps.numpy(), list(input_tensor.shape[1:]), 'cubic')

    handle1.remove()
    handle2.remove()

    if visualize:
        plt.figure(figsize = (7, 7))

        plt.subplot(2, 2, 1)
        plt.imshow(utils.image.get_plt_image(input_tensor))
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(utils.image.get_plt_image(heatmap))
        plt.title('GradCAM Heatmap')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(utils.image.get_plt_image(input_tensor))
        plt.imshow(utils.image.get_plt_image(heatmap), alpha = 0.5)
        plt.title('GradCAM Heatmap Overlay')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        heatmap = utils.image.normalize(utils.image.torch_to_numpy(input_tensor)) * np.expand_dims(activation_maps, -1)
        plt.imshow(utils.image.get_plt_image(heatmap))
        plt.title('GradCAM Mapped Inputs')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path)

        plt.show()
    
    return activation_maps, heatmap