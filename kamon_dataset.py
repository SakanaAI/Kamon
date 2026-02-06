"""Load the data and present it as a Dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import collections
import csv
import hashlib
import jaconv
import jsonlines
import random
import noising
import sys
import time
import torch

from absl import logging
from copy import deepcopy
from PIL import Image
from torchvision import transforms
from typing import Any, Dict, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))

_CACHE_VERSION = 1

ORIG_PARSED = f"{ROOT}/data/index_parsed_claude_all.jsonl"
ORIG_TRANSLATED = f"{ROOT}/data/index_parsed_claude_all_translated_claude.jsonl"
ORIG_DESCRIPTIONS = f"{ROOT}/data/descriptions.jsonl"
PARSED = ORIG_PARSED
TRANSLATED = ORIG_TRANSLATED
DESCRIPTIONS = ORIG_DESCRIPTIONS


def _dataset_cache_dir() -> str:
  return os.path.join(ROOT, ".cache", "kamon_dataset")


def _cache_enabled(cache: bool) -> bool:
  if not cache:
    return False
  env = os.environ.get("KAMON_DATASET_CACHE", "1").strip().lower()
  return env not in ("0", "false", "no", "off")


def _file_fingerprint(path: str):
  try:
    st = os.stat(path)
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)
  except OSError:
    return (os.path.abspath(path), None, None)


def _hash_key(key) -> str:
  return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]


def _pack_image(img: Image.Image) -> Dict[str, Any]:
  return {
    "mode": img.mode,
    "size": img.size,
    "data": img.tobytes(),
  }


def _unpack_image(packed: Dict[str, Any]) -> Image.Image:
  return Image.frombytes(packed["mode"], tuple(packed["size"]), packed["data"])


def _pack_metadata(metadata):
  packed = []
  for elt in metadata:
    new_elt = dict(elt)
    if "image" in new_elt:
      new_elt["image"] = _pack_image(new_elt["image"])
    packed.append(new_elt)
  return packed


def _unpack_metadata(metadata):
  unpacked = []
  for elt in metadata:
    new_elt = dict(elt)
    if "image" in new_elt:
      new_elt["image"] = _unpack_image(new_elt["image"])
    unpacked.append(new_elt)
  return unpacked


def _atomic_torch_save(obj: Any, path: str) -> None:
  tmp_path = f"{path}.tmp.{os.getpid()}"
  torch.save(obj, tmp_path)
  os.replace(tmp_path, path)


def _load_data() -> Dict[str, Any]:
  parsed = {}
  with jsonlines.open(PARSED) as reader:
    for elt in reader:
      description = elt["description"].strip()
      parsed[description] = [
        jaconv.kata2hira(e["expr"]) for e in elt["analysis"]
      ]
  translations = {}
  with jsonlines.open(TRANSLATED) as reader:
    for elt in reader:
      description = elt["description"].strip()
      translations[description] = elt["translation"]
  with jsonlines.open(DESCRIPTIONS) as reader:
    data = []
    for elt in reader:
      description = elt["description"].strip()
      if description in parsed:
        elt["parsed"] = parsed[description]
        elt["description"] = jaconv.kata2hira(description).strip()
        if description in translations:
          elt["translation"] = translations[description]
        else:
          elt["translation"] = "NA"
        data.append(elt)
  return data


ALLDATA = _load_data()


def reload_data(parsed: str, translated: str, descriptions: str):
  global PARSED
  global TRANSLATED
  global DESCRIPTIONS
  global ALLDATA
  PARSED = parsed
  TRANSLATED = translated
  DESCRIPTIONS = descriptions
  logging.info("Reloading data:")
  logging.info(f"PARSED = {parsed}")
  logging.info(f"TRANSLATED = {translated}")
  logging.info(f"DESCRIPTIONS = {descriptions}")
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


def _extend_label_set(
    expr_to_label: Tuple[Dict[str, int]]
) -> Tuple[Dict[str, int], Dict[int, str]]:
  label_to_expr = {i: e for e, i in expr_to_label.items()}
  expressions = set()
  for elt in ALLDATA:
    for expr in elt["parsed"]:
      if expr not in expr_to_label:
        expressions.add(expr)
  label = max(label_to_expr.keys()) + 1
  for expr in expressions:
    expr_to_label[expr] = label
    label_to_expr[label] = expr
    label += 1
  return expr_to_label, label_to_expr


def _retrieve_image(path: str, size: int) -> Image:
  size = (size, size)
  img = Image.open(os.path.join(ROOT, path)).resize(size)
  if img.mode in ["RGBA", "LA"]:
    img.load()
    canvas = Image.new("RGB", img.size, (255, 255, 255))
    idx = 3 if img.mode == "RGBA" else 1
    canvas.paste(img, mask=img.split()[idx])
    img = canvas
  return img


class KamonDataset(torch.utils.data.Dataset):
  """Kamon dataset as a torch.utils.data.Dataset.

  Args:
    image_size: Size of image, defaulting to 224x224
    division: one of "train", "val", "test"
    dataset_mean: dataset mean for image normalization
    dataset_std: dataset STD for image normalization
    one_hot: whether to present the text tensor as one_hot or not
    omit_edo: whether to omit the Edo data, which are rather different
    omit_from_test_val: data subsets to omit from test data
    pad: if True, pad to max length of all data.
    expr_to_label: If provided, create label set by extending this one
    cache: if True, cache the dataset to disk for faster loading
  """

  def __init__(
      self,
      image_size: int=224,
      division: str="train",
      dataset_mean: list=[0.5, 0.5, 0.5],
      dataset_std: list=[0.5, 0.5, 0.5],
      one_hot: bool=False,
      omit_edo: bool=False,
      omit_from_test_val: list[str]=[],
      pad: bool=True,
      num_augmentations: int=5,
      expr_to_label: Tuple[Dict[str, int]]=None,
      cache: bool=True,
  ):
    assert division in ["train", "val", "test"]
    self.image_size = image_size
    self.bigrams = collections.defaultdict(set)

    cache_path = ""
    if expr_to_label is None and _cache_enabled(cache):
      omit_from_test_val_key = tuple(sorted(omit_from_test_val))
      aug_key = num_augmentations if division == "train" else 0
      cache_key = (
        _CACHE_VERSION,
        division,
        image_size,
        omit_edo,
        omit_from_test_val_key,
        aug_key,
        _file_fingerprint(PARSED),
        _file_fingerprint(TRANSLATED),
        _file_fingerprint(DESCRIPTIONS),
        _file_fingerprint(os.path.join(ROOT, "noising.py")),
      )
      cache_dir = _dataset_cache_dir()
      try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"dataset_{division}_{_hash_key(cache_key)}.pt")
      except OSError:
        cache_path = ""

    if cache_path and os.path.exists(cache_path):
      try:
        cached = torch.load(cache_path)
        self.expr_to_label = cached["expr_to_label"]
        self.label_to_expr = cached["label_to_expr"]
        self.max_v = cached["max_v"]
        self.end_token = cached["end_token"]
        self.vocab_size = cached["vocab_size"]
        self.max_len = cached["max_len"]
        self.all_metadata = _unpack_metadata(cached["all_metadata"])
        self.metadata = _unpack_metadata(cached["metadata"])

        # Prepare image transform + label formatting (same as non-cached path)
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
        if division == "train":
          self._create_bigram_table()
        logging.info(f"Loaded dataset cache: {cache_path}")
        return
      except Exception as e:
        logging.info(f"Failed to load dataset cache ({cache_path}), regenerating: {e}")

    self.all_metadata = []
    data = ALLDATA
    if expr_to_label:
      self.expr_to_label, self.label_to_expr = _extend_label_set(expr_to_label)
    else:
      self.expr_to_label, self.label_to_expr = _create_label_set()
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
            "translation": elt["translation"],
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
      self._create_bigram_table()
      random.seed(time.time())
      new_train = []
      for elt in self.metadata:
        for _ in range(num_augmentations):
          new_elt = deepcopy(elt)
          new_elt["image"] = noising.apply_adjustments(new_elt["image"])
          new_train.append(new_elt)
      self.metadata += new_train
      random.shuffle(self.metadata)
    elif division == "val":
      metadata = self.all_metadata[train_top:val_top]
      metadata = [e for e in metadata if e["source"] not in omit_from_test_val]
      self.metadata = metadata
    else:
      metadata = self.all_metadata[val_top:]
      metadata = [e for e in metadata if e["source"] not in omit_from_test_val]
      self.metadata = metadata
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

    if cache_path:
      try:
        cached = {
          "expr_to_label": self.expr_to_label,
          "label_to_expr": self.label_to_expr,
          "max_v": self.max_v,
          "end_token": self.end_token,
          "vocab_size": self.vocab_size,
          "max_len": self.max_len,
          "all_metadata": _pack_metadata(self.all_metadata),
          "metadata": _pack_metadata(self.metadata),
        }
        _atomic_torch_save(cached, cache_path)
        logging.info(f"Wrote dataset cache: {cache_path}")
      except Exception as e:
        logging.info(f"Failed to write dataset cache ({cache_path}): {e}")

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    item = self.metadata[idx]
    image = item["image"]
    labels = item["labels"]
    if self.pad:
      labels = torch.tensor((labels + self.padded)[:self.max_len])
    if self.one_hot:
      labels = torch.nn.functional.one_hot(labels, self.max_v)
    return (
      self.transform(image),
      labels,
    )

  def dump_text(self, path: str):
    """Dumps partition in a text format, one string per line.
    """
    seen = set()
    corpus = []
    for elt in self.metadata:
      text = " ".join([self.label_to_expr[t] for t in elt["labels"]])
      seen.add(text)
    with open(path, "w") as s:
      for text in seen:
        s.write(f"{text}\n")

  # Create a bigram table. This is created for the train division only, and
  # *before* any metadata augmentations.
  def _create_bigram_table(self):
    for t in self.metadata:
      prev = -1
      for l in t["labels"]:
        self.bigrams[prev].add(l)
        prev = l
    self.bigrams[self.end_token].add(self.end_token)
