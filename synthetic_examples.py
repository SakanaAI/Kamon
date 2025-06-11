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
from PIL import Image

NUM = flags.DEFINE_integer("num", 10, "How many crests to generate")
GENERATED = flags.DEFINE_string(
  "generated",
  "synthetic",
  "Subdirectory for generated synthetic data",
)


MODS = ["", "覗き", "豆", "三つ盛り", "尻合せ三つ", "頭合せ三つ"]


def generate():
  num_containers = random.choice([1, 2])
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
  if modifier == "覗き":
    img = io.inside(container[1], other[1], peek=True)
    expr = f"{container[0]}に覗き{other[0]}"
  elif modifier == "豆":
    img = io.inside(container[1], other[1], peek=False, scale=io.BEAN)
    expr = f"{container[0]}に豆{other[0]}"
  elif modifier == "三つ盛り":
    img = io.stack(other[1], mode=io.StackMode.BASIC)
    img = io.inside(container[1], img, peek=False)
    expr = f"{container[0]}に三つ盛り{other[0]}"
  elif modifier == "尻合せ三つ":
    img = io.stack(other[1], mode=io.StackMode.SHIRI)
    img = io.inside(container[1], img, peek=False)
    expr = f"{container[0]}に尻合せ三つ{other[0]}"
  elif modifier == "頭合せ三つ":
    img = io.stack(other[1], mode=io.StackMode.ATAMA)
    img = io.inside(container[1], img, peek=False)
    expr = f"{container[0]}に頭合せ三つ{other[0]}"
  else:
    img = io.inside(container[1], other[1], peek=False)
    expr = f"{container[0]}に{other[0]}"
  idx -= 1
  while idx > -1:
    container = containers[idx]
    img = io.inside(container[1], img, peek=False)
    expr = f"{container[0]}に{expr}"
    idx -= 1
  # remove spaces:
  expr = "".join(expr.split())
  return img, expr, main_container_term, final_term


def main(unused_argv):
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
