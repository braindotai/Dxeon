from torchvision import transforms
import dxeon as dx
from torchvision.models import resnet18, mnasnet1_0

image = dx.io.image.read_pil('tests/assets/lion.jpg')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])
input_tensor = transform(image)

model = resnet18(pretrained = True)
model.eval()

dx.interpretability.compute_saliency_maps(model, input_tensor)