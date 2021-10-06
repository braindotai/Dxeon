import torch
import torchvision
from dxeon import io, interpretability

model = torchvision.models.resnet18(pretrained = True)
model.eval()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

image = transform(io.image.read_pil('dxeon/apple.jpg'))

ig = interpretability.compute_integrated_gradients(model, image, visualize = True)