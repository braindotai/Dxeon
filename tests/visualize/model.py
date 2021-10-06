import torch
from dxeon import visualize, stats
import torchvision

model = torchvision.models.resnet18().cuda()
# stats.model.summary(model, (4, 3, 224, 224))
visualize.model(model, input_size = (4, 3, 224, 224))
x = torch.randn(5, 5, 5, 5)

stats.summarize(model)

stats.summarize(x)