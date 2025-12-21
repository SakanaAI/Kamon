#!/bin/bash
# Sample inference script.
CKPTDIR=checkpoints
OUTDIR=outputs
CKPT=${CKPTDIR}/checkpoint_best_*.pt
echo Checkpoint is ${CKPT}
python3 inference.py \
        --checkpoint_path ${CKPT} \
        --dataset_subset=test \
        --omit_edo \
        --output_file="${OUTDIR}/test_decode.jsonl"
