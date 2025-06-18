#!/bin/bash
TRAINING_ITER=1_000_001
BATCH_SIZE=32
LR=1e-4
LR=1e-3
python3 \
    -m tasks.kamon.train \
    --d_model 2048 \
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
    --dropout 0.1 \
    --no-do_normalisation \
    --positional_embedding_type none \
    --backbone_type resnet34-2 \
    --batch_size "${BATCH_SIZE}" \
    --batch_size_test "${BATCH_SIZE}" \
    --lr "${LR}" \
    --training_iterations "${TRAINING_ITER}" \
    --warmup_steps 10000 \
    --use_scheduler \
    --scheduler_type cosine \
    --weight_decay 0.0 \
    --log_dir logs/kamon/run1 \
    --save_every 2000 \
    --track_every 5000 \
    --seed 42 \
    --n_test_batches 50 \
    --cirriculum_lookahead 3 \
    --device 0
