"""Various image operations.

TODO(rws): Filter this for stuff we actually need.
"""

import csv
import numpy as np
import random

from collections import defaultdict
from copy import deepcopy
from enum import Enum
from PIL import Image
from typing import Dict

BEAN = 0.25
NORMAL = 0.55
STACK_SCALE = 0.4
WHITE = 255, 255, 255
BLACK = 0, 0, 0


Mask = list[list[int], list[int], list[int]]


def image_to_array(img: Image) -> np.array:
  return np.array(img.getdata()).reshape((img.height, img.width, -1))


def array_to_image(ar: np.array) -> Image:
  return Image.fromarray(ar.astype(np.uint8))


def create_nonwhite_masks(
    ar: np.array,
    shape: tuple[int, int, int],
    offset: tuple[int, int],
) -> Dict[int, Mask]:
  canvas = np.ones(shape) * 255
  canvas[
    offset[0]:offset[0] + ar.shape[0],
    offset[1]:offset[1] + ar.shape[1],
    :,
  ] = ar
  masks = defaultdict(Mask)
  for h in range(shape[0]):
    for w in range(shape[1]):
      for d in range(shape[2]):
        pix = canvas[h, w, d]
        if pix not in masks:
          masks[pix] = [], [], []
        if pix != 255:
          masks[pix][0].append(h)
          masks[pix][1].append(w)
          masks[pix][2].append(d)
  return masks


# TODO(rws): Need various size checks for this:
def apply_nonwhite_masks(ar: np.array, masks: Dict[int, Mask]):
  ar = deepcopy(ar)
  for pix in masks:
    ar[masks[pix]] = pix
  return ar


def _exterior_masks(ar: np.array) -> tuple[Mask, Mask]:
  """Computes the exterior (= non-interior) maskf for a ring.

  Args:
    ar: an array
  Returns:
    A tuple of black and white masks.
  """
  assert len(ar.shape) == 3
  h, w = ar.shape[:2]
  interior = set()
  mid = int(w / 2)
  for i in range(h):
    j = mid
    while j >= 0:
      if np.array_equal(ar[i, j, :], [0, 0, 0]):
        break
      interior.add((i, j))
      j -= 1
    j = mid + 1
    while j < w:
      if np.array_equal(ar[i, j, :], [0, 0, 0]):
        break
      interior.add((i, j))
      j += 1
  blackhs = []
  blackws = []
  blackds = []
  whitehs = []
  whitews = []
  whiteds = []
  for i in range(h):
    for j in range(w):
      if (i, j) in interior:
        continue
      for k in range(3):
        if ar[i, j, k] == 255:
          whitehs.append(i)
          whitews.append(j)
          whiteds.append(k)
        # NB: Poor man's solution since this doesn't allow for grays.
        else:
          blackhs.append(i)
          blackws.append(j)
          blackds.append(k)
  bmask = blackhs, blackws, blackds
  wmask = whitehs, whitews, whiteds
  return bmask, wmask


def inside(
    a: Image,
    b: Image,
    scale=NORMAL,
    peek=False,
) -> Image:
  """Put resized image b inside image a.

  Args:
    a: an Image
    b: an Image
    scale: a float
    peek: if True, make "peeking" version
  Returns:
    An Image
  """
  b = b.resize((int(b.height * scale), int(b.width * scale)))
  adata = image_to_array(a)
  bdata = image_to_array(b)
  h = int((adata.shape[0] - bdata.shape[0]) / 2)
  w = int((adata.shape[1] - bdata.shape[1]) / 2)
  if peek:
    h += int(bdata.shape[0] * 0.5)
    over = h + bdata.shape[0] - adata.shape[0]
    if over > 0:
      h -= over
  masks = create_nonwhite_masks(bdata, shape=adata.shape, offset=(h, w))
  exterior_bmask, exterior_wmask = _exterior_masks(adata)
  adata = apply_nonwhite_masks(adata, masks)
  adata[exterior_bmask] = 0
  adata[exterior_wmask] = 255
  return array_to_image(adata)


def safe_rotate(img: Image, angle: int, pad: int=20) -> Image:
  ar = image_to_array(img)
  shape = ar.shape[0] + 2 * pad, ar.shape[1] + 2 * pad, ar.shape[-1]
  canvas = np.ones(shape) * 255
  canvas[
    pad:pad + ar.shape[0],
    pad:pad + ar.shape[1],
    :,
  ] = ar
  return array_to_image(canvas).rotate(angle, fillcolor=WHITE)


class StackMode(Enum):
  BASIC = 1
  SHIRI = 2
  ATAMA = 3


def stack(img: Image, mode: StackMode=StackMode.BASIC) -> Image:
  """Stack an image (盛り).

  The positioning here is very fiddly and depends in part on the pad value
  passed to safe_rotate. This is not general at all and assumes a base image
  size of 224x224.

  Args:
    img: Image
    mode: Optional mode, must be one of "atama", "shiri".
  Returns:
    An Image.
  """
  h, w = img.size
  sh, sw = int(h * STACK_SCALE), int(w * STACK_SCALE)
  small_img = img.resize((sh, sw))
  ar = image_to_array(img)
  sar = image_to_array(small_img)
  sw_half = int(sw * .5)
  left = int(w * .5) - sw_half
  sar1 = sar
  rotpad = 20

  def set_points(which):
    left = top = 0
    if which == "top":
      left = int(w * .5) - sw_half
      if mode == StackMode.ATAMA:
        top = int(h * .09)
      else:
        top = int(h * .05)
    elif which == "bottom_left":
      if mode == StackMode.ATAMA:
        top = int(h * .3)
        left = int(w * .28) - sw_half - 10
      elif mode == StackMode.SHIRI:
        top = int(h * .35)
        left = int(w * .28) - sw_half - 18
      else:
        top = int(h * .45)
        left = int(w * .28) - sw_half
    elif which == "bottom_right":
      if mode == StackMode.ATAMA:
        top = int(h * .3)
        left = int(w * .72) - sw_half - rotpad - 10
      elif mode == StackMode.SHIRI:
        top = int(h * .35)
        left = int(w * .72) - sw_half - rotpad - 2
      else:
        top = int(h * .45)
        left = int(w * .72) - sw_half
    return top, left

  top, left = set_points("top")
  if mode == StackMode.ATAMA:
    sar1 = image_to_array(small_img.rotate(180, fillcolor=WHITE))
  masks1 = create_nonwhite_masks(
    sar1,
    shape=ar.shape,
    offset=(top, left),
  )
  if mode == StackMode.ATAMA:
    sar2 = image_to_array(safe_rotate(small_img, 300, pad=rotpad))
  elif mode == StackMode.SHIRI:
    sar2 = image_to_array(safe_rotate(small_img, 120, pad=rotpad))
  else:  # StackMode.BASIC
    sar2 = sar
  top, left = set_points("bottom_left")
  masks2 = create_nonwhite_masks(
    sar2,
    shape=ar.shape,
    offset=(top, left),
  )
  sar3 = sar
  if mode == StackMode.ATAMA:
    sar3 = image_to_array(safe_rotate(small_img, 60, pad=rotpad))
  elif mode == StackMode.SHIRI:
    sar3 = image_to_array(safe_rotate(small_img, 240, pad=rotpad))
  else:  # StackMode.BASIC
    sar3 = sar
  top, left = set_points("bottom_right")
  masks3 = create_nonwhite_masks(
    sar3,
    shape=ar.shape,
    offset=(top, left),
  )
  canvas = np.ones_like(ar) * 255
  canvas = apply_nonwhite_masks(canvas, masks1)
  canvas = apply_nonwhite_masks(canvas, masks2)
  canvas = apply_nonwhite_masks(canvas, masks3)
  return array_to_image(canvas)


def _load_image_data():
  containers = {}
  others = {}
  with open("basic_charges.csv") as stream:
    reader = csv.reader(stream, delimiter=",", quotechar='"')
    for row in reader:
      if row[2] == "container":
        containers[row[1]] = row[0]
      else:
        others[row[1]] = row[0]
  return containers, others


CONTAINERS, OTHERS = _load_image_data()
ALL = deepcopy(CONTAINERS)
ALL.update(OTHERS)


# Temporary
def parse_containment(expr, scale=NORMAL, scale_decr=0.9):
  expr = expr.split("に")
  if len(expr) == 1:
    if expr[0] in ALL:
      return Image.open(ALL[expr[0]])
    return None
  if expr[0] in CONTAINERS:
    img = Image.open(CONTAINERS[expr[0]])
  else:
    return None
  right_branching = [(img,  scale, False)]
  for e in expr[1:]:
    if e in ALL:
      right_branching.append((Image.open(ALL[e]), scale, False))
    elif e.startswith("覗き") and e[2:] in ALL:
      right_branching.append((Image.open(ALL[e[2:]]), scale, True))
    else:
      return None
    scale *= scale_decr
  nimg = len(right_branching)
  i = nimg - 1
  img = right_branching[i][0]
  scale = right_branching[i][1]
  peek = right_branching[i][2]
  i -= 1
  while i >= 0:
    img = inside(right_branching[i][0], img, scale=scale, peek=peek)
    scale = right_branching[i][1]
    peek = right_branching[i][2]
    i -= 1
  return img


def random_expression(scale=NORMAL):
  expression = []
  expression.append(random.choice(list(CONTAINERS.keys())))
  if random.random() < 0.5:
    expression.append(random.choice(list(CONTAINERS.keys())))
  expression.append(random.choice(list(OTHERS.keys())))
  expression = "に".join(expression)
  img = parse_containment(expression, scale)
  img.show()
  return expression
