# Dxeon

## IO

- A single api for reading images to cv2, pil and torch.
- A single api for reading images from url to cv2, pil and torch.
- A single api for writing images from cv2, pil and torch.

```python
from dxeon import io

pil_image = io.image.read_pil(image_path, size = [512, 256], interpolation = 'bicubic')
cv2_image = io.image.read_cv2(image_path, size = 512, interpolation = 'cubic')
torch_image = io.image.read_torch(image_path, size = [512, 512], interpolation = 'bilinear')

pil_image = io.image.from_url_to_pil(url, size = [512, 512])
cv2_image = io.image.from_url_to_cv2(url, size = [512, 512])
torch_image = io.image.from_url_to_torch(url, size = [512, 512])

io.image.write(pil_image, 'image_pil.jpg')
io.image.write(cv2_image, 'image_cv2.jpg')
io.image.write(torch_image, 'image_torch.jpg')
```

## Model interpretability

- Multiple model interpretation methods, usable by a single line api call.

```python
from dxeon import interpretability

interpretability.visualize_grad_cam(model, inputs, ...)
interpretability.visualize_grad_cam_plus_plus(model, inputs, ...)
interpretability.visualize_guided_backprop(model, inputs, ...)
interpretability.visualize_integrated_gradients(model, inputs, ...)
interpretability.visualize_guided_integrated_gradients(model, inputs, ...)
```

## Losses

- Multiple losses module

```python
from dxeon import losses

losses.acgan_loss()
losses.arcface_loss()
losses.cgan_loss()
losses.circle_loss()
losses.content_loss()
losses.gan_loss()
losses.cyclegan_loss()
losses.gradient_penalty_loss()
losses.hingegan_loss()
losses.lsgan_loss()
losses.ragan_loss()
losses.style_loss()
losses.arcface_loss()
losses.circle_loss()
losses.focal_loss()
losses.dice_loss()
losses.vae_loss()
losses.triplet_loss()
losses.gram_matrix_loss()

```

## Model architecture

- Multiple basic blocks from different architectures

```python
from dxeon import modules

modules.conv_block
modules.dense_block
modules.depthwise_separable_block
modules.inception_block
modules.minibatch_discrimination
modules.pixel_norm
modules.residual_block
modules.resnext_block
modules.squeeze_and_excitation_block
modules.weight_standardization_block
```

## Model and tensor statictics

```python
from dxeon import stats

stats.summarize(model, ...)
stats.summarize(tensor)
```

## Model visualization

```python
from dxeon import visualize

visualize.model(model, ...)
```

## Learning rate warmups

```python
from dxeon import warmups

warmups.Linear(optimizer, ...)
warmups.Polynomial(optimizer, ...)
warmups.Exponential(optimizer, ...)
```