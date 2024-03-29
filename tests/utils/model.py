import torch
from torch import nn
from torch._C import device
from torchvision import models, transforms
import dxeon as dx

model = models.resnet18(pretrained = True)
model.eval().cuda()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# image_name = 'lion'
image_name = 'deer'
# image_name = 'apple'
# image_name = 'pandas'

input_tensor = transform(dx.io.image.read_pil(f'tests/assets/{image_name}.jpg')).cuda()

dx.utils.benchmark_performance(model, torch.stack([input_tensor] * 32, dim = 0))

torch_output = model(input_tensor.unsqueeze(0))[0].argmax(0)
print(torch_output)

state_dict_path = 'tests/model.pth'
dx.utils.save_model(model, state_dict_path)

onnx_path = 'tests/model.onnx'
onnx_model = dx.utils.GenerateONNXModel(model, state_dict_path, onnx_path, [1, 3, 224, 224])
onnx_output = onnx_model(input_tensor.unsqueeze(0).cpu())[0].argmax(0)
print(onnx_output)

onnx_model = dx.utils.LoadONNXModel(onnx_path)
onnx_output = onnx_model(input_tensor.unsqueeze(0).cpu())[0].argmax(0)
print(onnx_output)