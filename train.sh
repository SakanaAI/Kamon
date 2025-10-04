#!/bin/bash
CKPTDIR=checkpoints
OUTDIR=outputs
mkdir -p "${CKPTDIR}" "${OUTDIR}"
python3 train.py \
        --num_epochs=200 \
        --checkpoint_steps=5_000 \
        --checkpoint_dir="${CKPTDIR}" \
        --output_dir="${OUTDIR}" \
        --also_train_vgg \
        --ngram_length=3
