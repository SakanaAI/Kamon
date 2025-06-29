import glob
import numpy as np
import random

from PIL import Image, ImageEnhance


# More or less the same as image_operations safe_rotate()
#
# The extra code to make sure that once it's rotated, we have an all-white
# background.
def rotate(img: Image, pad: int=2, window: int=10) -> Image:

  size = img.size

  def random_angle():
    return random.randint(-window, window)

  def image_to_array(img: Image) -> np.array:
    return np.array(img.getdata()).reshape((img.height, img.width, -1))

  def array_to_image(ar: np.array) -> Image:
    return Image.fromarray(ar.astype(np.uint8))

  WHITE = 255, 255, 255

  ar = image_to_array(img)
  shape = ar.shape[0] + 2 * pad, ar.shape[1] + 2 * pad, ar.shape[-1]
  canvas = np.ones(shape) * 255
  canvas[
    pad:pad + ar.shape[0],
    pad:pad + ar.shape[1],
    :,
  ] = ar
  new_img = array_to_image(canvas).rotate(random_angle(), fillcolor=WHITE)
  return new_img.resize(size)


def randrange(bot: float, top: float) -> float:
  assert top > bot
  return random.random() * (top - bot) + bot


def salt_and_pepper(img: Image, bot: float=0.002, top: float=0.006) -> Image:
  output = np.copy(np.array(img))
  amount = randrange(bot, top)
  # add salt
  nb_salt = int(np.ceil(amount * output.size * 0.5))
  coords = [
    (random.randint(0, output.shape[0] - 1),
     random.randint(0, output.shape[1] - 1))
    for _ in range(nb_salt)
  ]
  for x, y in coords:
    output[x, y] = 1
  # add pepper
  nb_pepper = int(np.ceil(amount * output.size * 0.5))
  coords = [
    (random.randint(0, output.shape[0] - 1),
     random.randint(0, output.shape[1] - 1))
    for _ in range(nb_pepper)
  ]
  for x, y in coords:
    output[x, y] = 0
  img = Image.fromarray(output)
  return img


def gaussian(img: Image, bot: float=0.1, top: float=0.3) -> Image:
  output = np.copy(np.array(img, dtype=np.float64))
  row, col, depth = output.shape
  mean = 0
  var = randrange(bot, top)
  sigma = var ** 0.5
  gauss = np.random.normal(mean, sigma, (row, col, depth))
  gauss = gauss.reshape(row, col, depth)
  output += gauss
  output = np.array(output, dtype=np.uint8)
  img = Image.fromarray(output)
  return img


def adjust_brightness(img: Image):
  var = random.random() + 0.5
  return ImageEnhance.Brightness(img).enhance(var)


def adjust_contrast(img: Image):
  var = random.random() + 0.5
  return ImageEnhance.Contrast(img).enhance(var)


ADJUSTMENTS = [
  rotate,
  salt_and_pepper,
  gaussian,
  adjust_brightness,
  adjust_contrast,
]


def _apply_adjustments(img: Image) -> Image:
  new_img = img
  for adjustment in ADJUSTMENTS:
    if random.random() < 0.5:
      new_img = adjustment(new_img)
  return new_img


# Apply set of adjustments to the image from above up to niter times, with prob
# of 0.5 each time:
def apply_adjustments(img: Image, niter=3) -> Image:
  new_img = _apply_adjustments(img)
  for _ in range(niter - 1):
    if random.random() < 0.5:
      new_img = _apply_adjustments(img)
  return new_img
