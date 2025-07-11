#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
echo "Currently visible CUDA capable devices: ${CUDA_VISIBLE_DEVICES}"

TIME=$((10*3600))
echo "Setting time limit to $(($TIME / 3600)) hours"
ulimit -t $TIME

INPUT_FOLDER="/home/xpetrus/DP/Datasets/TNT_data/annotations/splitannotations64/IMG"
OUTPUT_FOLDER="./output"
MASK_FOLDER="/home/xpetrus/DP/Datasets/TNT_data/annotations/splitannotations64/GT_MERGED_LABELS"
LOG_FOLDER="./output/logs"
EPOCHS=500
DEVICE="cuda"
WORKERS=4
BATCH=16
MLFLOW_SERVER_PORT=8000

nice -n 19 python training.py \
    --input_folder "${INPUT_FOLDER}" \
    --output_folder "${OUTPUT_FOLDER}" \
    --mask_folder "${MASK_FOLDER}" \
    --log_folder "${LOG_FOLDER}" \
    --epochs ${EPOCHS} \
    --device ${DEVICE} \
    --num_workers ${WORKERS} \
    --batch_size ${BATCH} \
    --mlflow_port ${MLFLOW_SERVER_PORT}


