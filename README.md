# Kamon

![Example
 Mon](https://github.com/SakanaAI/Kamon/blob/main/data/mon-white-224/29605.jpg)

立ち浪に真向き兎

Kamon (Mon --- Japanese Family Crest) data from three sources:

1. Edo period Ansei Bukan (Armory of the Ansei Reign Years)

2. Various open-source images from Wikimedia.

3. Images from https://github.com/Rebolforces/kamondataset

All of these are "blazoned" with descriptions following the standard methods for
describing Mon. With the exception of the Wikimedia data, which already came
with descriptions, all examples were blazoned by hand.

# Installation

`pip3 install -r requirements.txt`

# Usage

`kamon_dataset.py` contains a wrapper that presents the data as a
`torch.utils.data.Dataset`.
