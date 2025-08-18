#!/bin/bash
fc-cache -f -v ~/.local/share/fonts
# Make sure they are there
fc-list :lang=ja family
TRAINING_ITER=1_000_001
TRAINING_ITER=500_001
TRAINING_ITER=800_001
BATCH_SIZE=32
LR=1e-4  # <== This was the default
# First settings
LR=1e-3
DROPOUT=0.1
BACKBONE_TYPE=resnet34-2
IMAGE_SIZE=224
NIMAGE_DUMPS=10
LOG_DIR=logs/kamon/run1
TRACK_EVERY=1000
# Second/third settings
LR=1e-3
DROPOUT=0.5
BACKBONE_TYPE=resnet34-4
IMAGE_SIZE=128
NIMAGE_DUMPS=5
LOG_DIR=logs/kamon/run2
LOG_DIR=logs/kamon/run3
TRACK_EVERY=2000
D_MODEL=2048
# Fourth settings
LOG_DIR=logs/kamon/run4
D_MODEL=4096
# Fifth settings, no LM
LOG_DIR=logs/kamon/run5
D_MODEL=2048
# Sixth settings, no LM
LOG_DIR=logs/kamon/run6
D_MODEL=8192  # This is 228_157_502 parameters
#
python3 \
    -m tasks.kamon.train \
    --d_model "${D_MODEL}" \
    --d_input 512 \
    --synapse_depth 4 \
    --heads 8 \
    --n_synch_out 6 \
    --n_synch_action 32 \
    --neuron_select_type random-pairing \
    --iterations 75 \
    --memory_length 25 \
    --deep_memory \
    --memory_hidden_dims 32 \
    --dropout "${DROPOUT}" \
    --backbone_type "${BACKBONE_TYPE}" \
    --no-do_normalisation \
    --positional_embedding_type none \
    --batch_size "${BATCH_SIZE}" \
    --batch_size_test "${BATCH_SIZE}" \
    --lr "${LR}" \
    --training_iterations "${TRAINING_ITER}" \
    --warmup_steps 10000 \
    --use_scheduler \
    --scheduler_type cosine \
    --weight_decay 0.0 \
    --log_dir "${LOG_DIR}" \
    --save_every 2000 \
    --track_every "${TRACK_EVERY}" \
    --seed 42 \
    --n_test_batches 50 \
    --cirriculum_lookahead 3 \
    --nimage_dumps "${NIMAGE_DUMPS}" \
    --image_size "${IMAGE_SIZE}" \
    --device 0 \
    --reload \
    # --ngram_lm_scaling 20 \
    # --ngram_lm ../Work/Kamon/ngram/train_s.mod \
