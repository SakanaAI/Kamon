"""Load the data and present it as a Dataset.
"""

import collections
import csv
import jaconv
import jsonlines
import os
import random
import torch

from PIL import Image
from torchvision import transforms
from typing import Any, Dict, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_data() -> Dict[str, Any]:
  parsed = {}
  with jsonlines.open(f"{ROOT}/data/index_parsed_claude_all.jsonl") as reader:
    for elt in reader:
      parsed[elt["description"]] = [
        jaconv.kata2hira(e["expr"]) for e in elt["analysis"]
      ]
  with jsonlines.open(f"{ROOT}/data/descriptions.jsonl") as reader:
    data = []
    for elt in reader:
      if elt["description"] in parsed:
        elt["parsed"] = parsed[elt["description"]]
        elt["description"] = jaconv.kata2hira(elt["description"]).strip()
        data.append(elt)
  return data


ALLDATA = _load_data()
END_TOKEN = "<EOS>"


def _create_label_set() -> Tuple[Dict[str, int], Dict[int, str]]:
  expressions = set()
  for elt in ALLDATA:
    for expr in elt["parsed"]:
      expressions.add(expr)
  expressions = sorted(list(expressions))
  expressions.append(END_TOKEN)
  expr_to_label = {e: i for i, e in enumerate(expressions)}
  label_to_expr = {i: e for e, i in expr_to_label.items()}
  return expr_to_label, label_to_expr


def _retrieve_image(path: str, size: int) -> Image:
  size = (size, size)
  return Image.open(os.path.join(ROOT, path)).resize(size)


class KamonDataset(torch.utils.data.Dataset):
  """Kamon dataset as a torch.utils.data.Dataset.

  Args:
    image_size: Size of image, defaulting to 224x224
    division: one of "train", "val", "test"
    dataset_mean: dataset mean for image normalization
    dataset_std: dataset STD for image normalization
    one_hot: whether to present the text tensor as one_hot or not
    omit_edo: whether to omit the Edo data, which are rather different
    pad: if True, pad to max length of all data.
  """

  def __init__(
      self,
      image_size: int=224,
      division: str="train",
      dataset_mean: list=[0.5, 0.5, 0.5],
      dataset_std: list=[0.5, 0.5, 0.5],
      one_hot: bool=False,
      omit_edo: bool=False,
      pad: bool=True,
  ):
    assert division in ["train", "val", "test"]
    self.image_size = image_size
    self.all_metadata = []
    data = ALLDATA
    self.expr_to_label, self._label_to_expr = _create_label_set()
    self.max_v = len(self.expr_to_label)
    self.end_token = self.expr_to_label[END_TOKEN]
    self.vocab_size = self.end_token + 1
    self.max_len = -1
    for elt in data:
      description = elt["description"]
      labels = [self.expr_to_label[e] for e in elt["parsed"]] + [self.end_token]
      if len(labels) > self.max_len:
        self.max_len = len(labels)
      for img in elt["images"]:
        source = img["source"]
        if omit_edo and source == "edo":
          continue
        path = img["path"]
        self.all_metadata.append(
          {
            "description": description,
            "labels": labels,
            "path": path,
            "source": source,
            "image": _retrieve_image(
              os.path.join(path),
              self.image_size,
            ).convert("RGB"),
          }
        )
    length = len(self.all_metadata)
    random.seed(length)
    random.shuffle(self.all_metadata)
    train_top = int(0.8 * length)
    val_top = int(0.9 * length)
    if division == "train":
      self.metadata = self.all_metadata[:train_top]
    elif division == "val":
      self.metadata = self.all_metadata[train_top:val_top]
    else:
      self.metadata = self.all_metadata[val_top:]

    ## Prepare image
    ##
    ## mean and std are just copied from what I had for the Stable Diffusion
    ## training.
    self.dataset_mean = dataset_mean
    self.dataset_std = dataset_std
    self.transform = transforms.Compose(
      [
        transforms.Resize((self.image_size, self.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(self.dataset_mean, self.dataset_std),
      ]
    )
    self.one_hot = one_hot
    self.pad = pad
    self.padded = [self.end_token] * self.max_len

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    item = self.metadata[idx]
    image = item["image"]
    labels = item["labels"]
    if self.pad:
      labels = (labels + self.padded)[:self.max_len]
    if self.one_hot:
      labels = torch.nn.functional.one_hot(torch.tensor(labels), self.max_v)
    return (
      self.transform(image),
      torch.tensor(labels),
    )
