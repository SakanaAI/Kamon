# Kamon (家紋)

![Example
 Mon](https://github.com/SakanaAI/Kamon/blob/main/data/mon-white-224/29605.jpg)

立ち浪に真向き兎 ('frontwards facing rabbit in a standing wave')

This repository contains kamon (Japanese family crest) data from three sources:

1. Edo period Ansei Bukan (安政武鑑, Armory of the Ansei Reign Years) from the [Center for Open Data in the Humanities](https://codh.rois.ac.jp/).

2. Various open-source images from Wikimedia
(see [here](https://github.com/SakanaAI/Kamon/blob/main/data/wiki/wiki_licenses.csv) for licensing details).

3. Images from https://github.com/Rebolforces/kamondataset.

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

Kamon are theoretically open-ended since one can create new designs by combining
existing motifs (or even creating new motifs) in new ways. Some of the ways of
constructing new crests can be described by simple grammatical rules. Here we
provide a simple grammar-based generator to create synthetic examples (some more
plausible than others):

`python synthetic_examples.py --num=10`

See the examples in the `synthetic` subdirectory, for example:

![Synthetic Example
 Mon](https://github.com/SakanaAI/Kamon/blob/main/synthetic/synth_0002.png)

月輪に覗き尻合わせ三つ紅葉 ('Peeking butts-together three maple leaves in a moon ring')

# Training and inference with baseline VGG-based model

One challenge is to generate the description of a crest given an image of that
crest. Vision models are not particularly well tuned for this sort of data, and
there are some important differences between scene-to-text and this problem.
The motifs in kamon are usually highly stylized, so that to recognize, say, a
wave requires knowing what a typical kamon stylization of a wave looks
like. Motifs may be modified and arranged in various ways, and while these
modifications and arrangements are quite restricted, they also often require
some amount of reasoning. For example a motif such as a plant leaf may be
arranged three in a circle, with either the heads pointed to the center (頭合わ
せ) or the bottoms pointed to the center (尻合わせ). But `head` here means the
top of the motif as it would normally be displayed, and `bottom` the
reverse. This requires knowing for each motif what the typical display
arrangement is, which is not obvious from the geometry of the motif. This,
coupled with the fact that the dataset for Kamon is small, makes crest-to-text
conversion challenging.

A baseline model using
[VGG](https://huggingface.co/learn/computer-vision-course/en/unit2/cnns/vgg) is
provided.  The architecture is given schematically below:

![VGG model architecture](https://github.com/SakanaAI/Kamon/blob/main/vgg.jpg)

A training script can be found in `train.sh` and an inference script in `test.sh`.

A script for generating an HTML page visualizing the inference output can be found in `visualize.py`.

Decoding output on the test set from one training run can be seen
[here](https://htmlpreview.github.io/?https://github.com/SakanaAI/Kamon/blob/main/test_decode.html).
