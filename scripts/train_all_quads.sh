#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
echo "Currently visible CUDA capable devices: ${CUDA_VISIBLE_DEVICES}"

TIME=$((10*3600))
echo "Setting time limit to $(($TIME / 3600)) hours"
ulimit -t $TIME

INPUT_ROOT="/home/xpetrus/Desktop/DP/Datasets/TNT_data/annotations/2025-09-19"
OUTPUT_BASE="./output/output-train-all-quads"
EPOCHS=50
WORKERS=4
BATCH=32
MLFLOW_SERVER_PORT=8800
LR=0.0001
MODEL=anisotropicunet
SEED=42
MODEL_DEPTH=5
WEIGHT_DECAY=0.001
HORIZONTAL_KERNEL="3,3,3"
HORIZONTAL_PADDING="1,1,1"

QUADS=(quad1 quad2 quad3 quad4)

for i in "${!QUADS[@]}"; do
    QUAD=${QUADS[$i]}
    INPUT_FOLDER="${INPUT_FOLDER}/${QUAD}"
    TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
    OUTPUT_FOLDER=""
done
