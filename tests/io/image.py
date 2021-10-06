from PIL import Image
import matplotlib.pyplot as plt
from dxeon import io

image = io.image.read_cv2('tests/assets/apple.jpg', size = [512, 512], interpolation = 'linear')
assert image.shape[0] == 512 and image.shape[1] == 512
plt.imshow(image)
plt.show()

image = io.image.read_pil('tests/assets/apple.jpg', size = [512, 512], interpolation = 'bilinear')
assert image.size[0] == 512 and image.size[1] == 512
image.show()

image = io.image.read_cv2('tests/assets/apple.jpg', size = 512, interpolation = 'cubic')
assert image.shape[1] == 512
plt.imshow(image)
plt.show()

image = io.image.read_pil('tests/assets/apple.jpg', size = 512, interpolation = 'bicubic')
assert image.size[0] == 512
image.show()

image = io.image.read_torch('tests/assets/apple.jpg', size = [512, 512], interpolation = 'bilinear')
print(image.shape, image.max(), image.min())
plt.imshow(image.permute(1, 2, 0))
plt.show()

image = io.image.from_url_to_pil('https://image.shutterstock.com/image-photo/mountains-under-mist-morning-amazing-260nw-1725825019.jpg', size = [512, 512])
io.image.write(image, './tests/image_pil.jpg')

image = io.image.from_url_to_cv2('https://image.shutterstock.com/image-photo/mountains-under-mist-morning-amazing-260nw-1725825019.jpg', size = [512, 512])
io.image.write(image, './tests/image_cv2.jpg')

image = io.image.from_url_to_torch('https://image.shutterstock.com/image-photo/mountains-under-mist-morning-amazing-260nw-1725825019.jpg', size = [512, 512])
io.image.write(image, './tests/image_torch.jpg')