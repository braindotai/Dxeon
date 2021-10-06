import torch
from dxeon import visualize
from dxeon import io


image = io.image.read_torch('tests/assets/apple.jpg')
print(image.max(), image.min(), image.shape)

visualize.image(image)
visualize.image_stack([[image, image, image, image], [image, image, image, image]])
visualize.image_grid([image, image, image, image], (2, 2))
visualize.labeled_images([image] * 16, [1] * 16, (4, 4))
