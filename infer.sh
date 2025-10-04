#!/bin/bash
CKPTDIR=checkpoints
CKPT=${CKPTDIR}/checkpoint_best_*.pt
OUTDIR=outputs
echo Checkpoint is ${CKPT}
python3 inference.py \
        --checkpoint_path ${CKPT} \
        --dataset_subset=test \
        --omit_edo \
        --output_file="${OUTDIR}/test_decode.jsonl"
