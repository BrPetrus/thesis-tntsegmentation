#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
echo "Currently visible CUDA capable devices: ${CUDA_VISIBLE_DEVICES}"

TIME=$((10*3600))
echo "Setting time limit to $(($TIME / 3600)) hours"
ulimit -t $TIME

# INPUT_FOLDER="/home/xpetrus/Desktop/DP/Datasets/TNT_data/annotations/splitannotations90v2/" 
# INPUT_FOLDER="/home/xpetrus/DP/Datasets/TNT_data/annotations/2025-08-21"
# INPUT_FOLDER="/home/xpetrus/DP/Datasets/TNT_data/annotations/2025-08-21-quad2/"
INPUT_FOLDER="/home/xpetrus/Desktop/DP/Datasets/TNT_data/annotations/2025-08-21-quad2-norandomtest"
# INPUT_FOLDER="/home/xpetrus/Desktop/DP/Datasets/TNT_data/annotations/2025-08-21-quad2-norandom"
OUTPUT_FOLDER="./output-training"
EPOCHS=500
WORKERS=4
BATCH=16
MLFLOW_SERVER_PORT=8000
LR=0.001
SEED=42
WEIGHT_DECAY=1e-5

nice -n 19 python training.py \
    "${INPUT_FOLDER}" \
    "${OUTPUT_FOLDER}" \
    --epochs ${EPOCHS} \
    --num_workers ${WORKERS} \
    --batch_size ${BATCH} \
    --mlflow_port ${MLFLOW_SERVER_PORT} \
    --lr ${LR} \
    --seed ${SEED}
    --weight_decay ${WEIGHT_DECAY}


