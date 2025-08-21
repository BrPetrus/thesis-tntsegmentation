#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
echo "Currently visible CUDA capable devices: ${CUDA_VISIBLE_DEVICES}"

TIME=$((10*3600))
echo "Setting time limit to $(($TIME / 3600)) hours"
ulimit -t $TIME

# INPUT_FOLDER="/home/xpetrus/Desktop/DP/Datasets/TNT_data/annotations/splitannotations90v2/" 
INPUT_FOLDER="/home/xpetrus/DP/Datasets/TNT_data/annotations/2025-08-21"
OUTPUT_FOLDER="./output-training"
EPOCHS=250
WORKERS=4
BATCH=16
MLFLOW_SERVER_PORT=8000
LR=0.05

nice -n 19 python training.py \
    "${INPUT_FOLDER}" \
    "${OUTPUT_FOLDER}" \
    --epochs ${EPOCHS} \
    --num_workers ${WORKERS} \
    --batch_size ${BATCH} \
    --mlflow_port ${MLFLOW_SERVER_PORT} \
    --lr ${LR}


