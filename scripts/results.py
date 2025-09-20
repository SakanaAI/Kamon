import os
import sys
sys.path.append(os.path.dirname(".."))

import collections
import kamon_dataset as kd
import jsonlines


from absl import app
from absl import flags


RESULTS = flags.DEFINE_string("results", None, "Path to results JSONL")
OUTPUT = flags.DEFINE_string("output", None, "Path to output JSONL")


def fix(desc):
  return "".join(desc.strip().split())


def main(unused_argv):
  train = kd.KamonDataset(
    division="train",
    image_size=224,
    num_augmentations=0,
    one_hot=False,
    omit_edo=True,
  )
  train_images = collections.defaultdict(list)
  for elt in train.metadata:
    train_images[fix(elt["description"])].append(elt["path"])
  test = kd.KamonDataset(
    division="test",
    image_size=224,
    num_augmentations=0,
    one_hot=False,
    omit_edo=True,
  )
  results = [e for e in jsonlines.open(RESULTS.value)]
  merged = []
  for i, elt in enumerate(results):
    test_elt = test.metadata[i]
    merged_elt = {
      "reference": fix(elt["reference_description"]),
      "predicted": fix(elt["predicted_description"]),
      "image": test_elt["path"],
      "train_images_reference": train_images[fix(elt["reference_description"])],
      "train_images_predicted": train_images[fix(elt["predicted_description"])],
      "translation": test_elt["translation"],
    }
    merged.append(merged_elt)
  with jsonlines.open(OUTPUT.value, "w") as writer:
    writer.write_all(merged)


if __name__ == "__main__":
  flags.mark_flag_as_required("results")
  flags.mark_flag_as_required("output")
  app.run(main)
