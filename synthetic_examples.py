"""Construct new crests from containers and basic charges.

Currently supported:

1. Containment (XにY)
2. Peeking (覗き)
3. Miniature (豆)
4. Stacked (三つ盛り)
5. Stacked, rears together (尻合せ三つ)
6. Stacked, heads togehter (頭合せ三つ)

"""

import collections
import image_operations as io
import jsonlines
import os
import random

from absl import app
from absl import flags
from absl import logging
from PIL import Image

NUM = flags.DEFINE_integer("num", 10, "How many crests to generate")
GENERATED = flags.DEFINE_string(
  "generated",
  "synthetic",
  "Subdirectory for generated synthetic data",
)
MAX_CONTAINERS = flags.DEFINE_integer(
  "max_containers",
  2,
  "Maximum container nesting",
)
VANILLA_MOD_MULTIPLIER = flags.DEFINE_integer(
  "vanilla_mod_multiplier",
  1,
  "Weight the vanilla mod (no transformation) so that it is chosen more or "
  "less frequently",
)


MODS = ["覗き", "豆", "三つ盛り", "尻合せ三つ", "頭合せ三つ"]


def generate(keep_intermediate: bool=False):
  cont = list(range(1, MAX_CONTAINERS.value + 1))
  num_containers = random.choice(cont)
  container_keys = list(io.CONTAINERS.keys())
  containers = []
  for i in range(num_containers):
    key = random.choice(container_keys)
    containers.append((key, Image.open(io.CONTAINERS[key])))
  main_container_term = containers[0][0]
  other_keys = list(io.OTHERS.keys())
  key = random.choice(other_keys)
  final_term = key
  other = key, Image.open(io.OTHERS[key])
  modifier = random.choice(MODS)
  idx = len(containers) - 1
  container = containers[idx]
  intermediates = []

  def add_intermediate(img, expr, prepend=False):
    if keep_intermediate:
      if prepend:
        intermediates.insert(0, {"img": img, "expr": expr})
      else:
        intermediates.append({"img": img, "expr": expr})

  if modifier == "覗き":
    add_intermediate(container[1], container[0])
    add_intermediate(None, "に 覗き")
    add_intermediate(other[1], other[0])
    img = io.inside(container[1], other[1], peek=True)
    expr = f"{container[0]}に覗き{other[0]}"
  elif modifier == "豆":
    # Move this to here so we can add the reduced size to the intermediates if
    # needed.
    h, w = other[1].height, other[1].width
    img = other[1].resize((int(h * io.BEAN), int(w * io.BEAN)))
    add_intermediate(container[1], container[0])
    add_intermediate(img, f"に 豆 {other[0]}")
    add_intermediate(other[1], other[0])
    img = io.inside(container[1], img, peek=False)
    expr = f"{container[0]}に豆{other[0]}"
  elif modifier == "三つ盛り":
    img = io.stack(other[1], mode=io.StackMode.BASIC)
    add_intermediate(container[1], container[0])
    add_intermediate(img, f"に 三つ盛り {other[0]}")
    add_intermediate(other[1], other[0])
    img = io.inside(container[1], img, peek=False)
    expr = f"{container[0]}に三つ盛り{other[0]}"
  elif modifier == "尻合せ三つ":
    img = io.stack(other[1], mode=io.StackMode.SHIRI)
    add_intermediate(container[1], container[0])
    add_intermediate(img, f"に 尻合せ三つ {other[0]}")
    add_intermediate(other[1], other[0])
    img = io.inside(container[1], img, peek=False)
    expr = f"{container[0]}に尻合せ三つ{other[0]}"
  elif modifier == "頭合せ三つ":
    img = io.stack(other[1], mode=io.StackMode.ATAMA)
    add_intermediate(container[1], container[0])
    add_intermediate(img, f"に 頭合せ三つ {other[0]}")
    add_intermediate(other[1], other[0])
    img = io.inside(container[1], img, peek=False)
    expr = f"{container[0]}に頭合せ三つ{other[0]}"
  else:
    add_intermediate(container[1], container[0])
    add_intermediate(other[1], other[0])
    img = io.inside(container[1], other[1], peek=False)
    expr = f"{container[0]}に{other[0]}"
  idx -= 1
  while idx > -1:
    container = containers[idx]
    add_intermediate(img, f"に {expr}", prepend=True)
    add_intermediate(container[1], container[0], prepend=True)
    img = io.inside(container[1], img, peek=False)
    expr = f"{container[0]}に{expr}"
    idx -= 1
  # remove spaces:
  expr = "".join(expr.split())
  tup = img, expr, main_container_term, final_term
  if keep_intermediate:
    tup = tup + (intermediates,)
  return tup


def main(unused_argv):
  global MODS
  MODS = [""] * VANILLA_MOD_MULTIPLIER.value + MODS
  os.makedirs(GENERATED.value, exist_ok=True)
  data = collections.defaultdict(list)
  for i in range(NUM.value):
    img, expr, main_container_term, final_term = generate()
    path = f"{GENERATED.value}/synth_{i:04d}.png"
    img.save(path)
    data[expr].append(
      {
        "path": path,
        "source": "synthetic",
      }
    )
    msg = f"Added `{path}` for `{expr}` ({i}/{NUM.value})"
    logging.info(msg)
  jsonl = []
  for expr in data:
    jsonl.append(
      {
        "description": expr,
        "images": data[expr],
      }
    )
  path = os.path.join(GENERATED.value, "synthetic.jsonl")
  with jsonlines.open(path, "w") as writer:
    writer.write_all(jsonl)


if __name__ == "__main__":
  app.run(main)
