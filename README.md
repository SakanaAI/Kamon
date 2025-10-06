# Kamon (家紋)

![Example
 Mon](https://github.com/SakanaAI/Kamon/blob/main/data/mon-white-224/29605.jpg)

立ち浪に真向き兎 ('frontwards facing rabbit in a standing wave')

This repository contains kamon (Japanese family crest) data from three sources:

1. Edo period Ansei Bukan (安政武鑑, Armory of the Ansei Reign Years) from the [Center for Open Data in the Humanities](https://codh.rois.ac.jp/).

2. Various open-source images from Wikimedia
(see [here](https://github.com/SakanaAI/Kamon/blob/main/data/wiki/wiki_licenses.csv) for licensing details).

3. Images from https://github.com/Rebolforces/kamondataset. Note that since we are uncertain about the copyright
status of these data, we do not provide these images directly. Instead, please navigate to that site,
download the [tarball](https://github.com/Rebolforces/kamondataset/commits/main/mon-white-224.tar.gz), and install all the
images in the subdirectory `train` directly under `data/mon-white-224` here.

All of these are "blazoned" with descriptions following the standard methods for
describing kamon. With the exception of the Wikimedia data, which already came
with descriptions, all examples were blazoned by hand.

Total dataset size is 7,410 images paired with descriptions.  The data includes
(machine-generated) dependency parses and English translations for all
descriptions.

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


The dependency parsing of the crest descriptions, along with their translation
into English in `index_parsed_claude_all.jsonl` and
`index_parsed_claude_all_translated_claude.jsonl`, respectively, were performed
using Claude 3.5 Sonnet.

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

月輪に覗き尻合わせ三つ紅葉 ('Peeking bottoms-together three maple leaves in a moon ring')

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

The input image is replicated _N_ times, where _N_ is the maximum token length
of the output text. The image is optionally masked with a position-specific
mask, then passed to a VGG model that is shared across positions. The final
layer of the VGG model is removed so that the penultimate layer can be used for
features. The VGG model by default is set to be trainable.  The VGG's features
along with the `--ngram_length - 1` previous VGG features and the
`--ngram_length - 1` previous logits are input features for predicting the
logits for the current position. The intuition behind the masking is that since
the descriptions of the crests largely proceed from the outside inwards, the
model should focus on different parts of the image at different times, and thus
when considering the first output term it might learn to mask out bits of the
image that are usually less relevant for predicting that term. For example, many
crests are surrounded by some sort of ring or hexagon or other container, and
this is described first. In order to describe that, the motifs inside the
container are irrelevant. In practice it should be noted that the masking does
not (yet) seem to make much difference.

A training script (with masking on) can be found in `train.sh` and an inference
script in `test.sh`.

Decoding output on the test set from one training run can be seen
[here](https://github.com/SakanaAI/Kamon/blob/main/test_decode.jsonl).  Note that this training and evaluation
omits the Edo-period data.

A script for generating an HTML page visualizing the inference output can be found in `visualize_outputs.py`.
