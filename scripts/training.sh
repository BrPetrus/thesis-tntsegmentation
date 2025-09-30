#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
echo "Currently visible CUDA capable devices: ${CUDA_VISIBLE_DEVICES}"

TIME=$((10*3600))
echo "Setting time limit to $(($TIME / 3600)) hours"
ulimit -t $TIME

# INPUT_FOLDER="/home/xpetrus/Desktop/DP/Datasets/TNT_data/annotations/splitannotations90v2/" 
# INPUT_FOLDER="/home/xpetrus/DP/Datasets/TNT_data/annotations/2025-08-21"
# INPUT_FOLDER="/home/xpetrus/DP/Datasets/TNT_data/annotations/2025-08-21-quad2/"
# INPUT_FOLDER="/home/xpetrus/Desktop/DP/Datasets/TNT_data/annotations/2025-08-21-quad1-norandomtest"
# INPUT_FOLDER="/home/xpetrus/DP/Datasets/TNT_data/annotations/2025-09-19/quad1/"
INPUT_FOLDER="/home/xpetrus/Desktop/DP/Datasets/TNT_data/annotations/2025-09-30/quad1/"
OUTPUT_FOLDER="./output-training-$(date +%Y-%m-%d_%H-%M-%S)"
EPOCHS=10
WORKERS=4
BATCH=32
MLFLOW_SERVER_PORT=8800
LR=0.0001
MODEL=anisotropicunet_csam
SEED=42
MODEL_DEPTH=5
WEIGHT_DECAY=0.0001
HORIZONTAL_KERNEL="3,3,3"
HORIZONTAL_PADDING="1,1,1"

nice -n 19 python training.py \
    "${INPUT_FOLDER}" \
    "${OUTPUT_FOLDER}" \
    --epochs "${EPOCHS}" \
    --num_workers "${WORKERS}" \
    --batch_size "${BATCH}" \
    --mlflow_port "${MLFLOW_SERVER_PORT}" \
    --lr "${LR}" \
    --model "${MODEL}" \
    --seed "${SEED}" \
    --model_depth "${MODEL_DEPTH}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --horizontal_kernel "${HORIZONTAL_KERNEL}" \
    --horizontal_padding "${HORIZONTAL_PADDING}"
    # --shuffle


