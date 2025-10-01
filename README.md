# Kamon

![Example
 Mon](https://github.com/SakanaAI/Kamon/blob/main/data/mon-white-224/29605.jpg)

立ち浪に真向き兎 ('frontwards facing rabbit in a standing wave')

Kamon (Mon --- Japanese Family Crest) data from three sources:

1. Edo period Ansei Bukan (Armory of the Ansei Reign Years) from the [Center for Open Data in the Humanities](https://codh.rois.ac.jp/)

2. Various open-source images from Wikimedia
(see [here](https://github.com/SakanaAI/Kamon/blob/main/data/wiki/wiki_licenses.csv) for licensing details).

3. Images from https://github.com/Rebolforces/kamondataset

All of these are "blazoned" with descriptions following the standard methods for
describing Mon. With the exception of the Wikimedia data, which already came
with descriptions, all examples were blazoned by hand.

Total dataset size is 7,410 images paired with descriptions.

# Installation

`pip3 install -r requirements.txt`

# Usage

`kamon_dataset.py` contains a wrapper that presents the data as a
`torch.utils.data.Dataset`.

For example, the following loads the validation set. Each entry maps from a tensor representing the image to a
sequence of vocabulary items corresponding to the phrase describing the crest.

    val = kamon_dataset.KamonDataset(division="val", one_hot=False)
    val[0]

    (tensor([[[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         ...,
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         ...,
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         ...,
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]]]), [49, 1366, 1252])

# Synthetic examples

Synthetic examples can be generated from a simple grammar as follows:

`python synthetic_examples.py --num=10`

See the examples in the `synthetic` subdirectory, for example:

![Synthetic Example
 Mon](https://github.com/SakanaAI/Kamon/blob/main/synthetic/synth_0002.png)

月輪に覗き尻合わせ三つ紅葉 ('Peeking butts-together three maple leaves in a moon ring')

# Training and inference with baseline VGG-based model

A baseline model using
[VGG](https://huggingface.co/learn/computer-vision-course/en/unit2/cnns/vgg) is
provided.  The architecture is given schematically below:

![VGG model architecture](https://github.com/SakanaAI/Kamon/blob/main/vgg.jpg)

A training script can be found in `train.sh` and an inference script in `test.sh`.

A script for generating an HTML page visualizing the inference output can be found in `visualize.py`.
