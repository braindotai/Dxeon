import torchvision
from dxeon import io, interpretability, viz

resnet18 = torchvision.models.resnet18(pretrained = True)
resnet18.eval()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

image = transform(io.image.read_pil('tests/assets/deer.jpg'))

ig = interpretability.compute_guided_integrated_gradients(resnet18, image, visualize = True)

# viz.image(ig, cmap = 'inferno', show = False, size = (5, 5))
# viz.image(image, cmap = 'inferno', alpha = 0.3)