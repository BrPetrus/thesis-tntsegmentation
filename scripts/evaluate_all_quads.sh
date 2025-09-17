#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
echo "Currently visible CUDA capable devices: ${CUDA_VISIBLE_DEVICES}"

MODEL_PATH="/home/xpetrus/DP/DP-WIP/scripts/output-training-2025-09-16_16-49-58/model_final.pth"
DATA_ROOT="/home/xpetrus/Desktop/DP/Datasets/TNT_data/evaluations_datasets"
OUTPUT_FOLDER="./experiment/"
DEVICE="cuda"
BATCH_SIZE=64

for QUAD in quad1 quad2 quad3 quad4
do
    echo "Evaluating $QUAD..."
    python evaluate_models.py \
        "${MODEL_PATH}" \
        "$DATA_ROOT/$QUAD" \
        "$OUTPUT_FOLDER" \
        --device "$DEVICE" \
        --batch_size $BATCH_SIZE
done