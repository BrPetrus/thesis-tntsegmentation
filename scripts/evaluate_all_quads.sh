#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
echo "Currently visible CUDA capable devices: ${CUDA_VISIBLE_DEVICES}"

# MODEL_PATH="/home/xpetrus/DP/DP-WIP/scripts/output-training-2025-09-16_16-49-58/model_final.pth"
# MODEL_PATH="/home/xpetrus/DP/DP-WIP/scripts/output-training-2025-09-17_17-00-16/model_final.pth"  #dashing-kite-31
MODEL_PATH="/home/xpetrus/DP/DP-WIP/scripts/output/output-train-all-quads/2025-09-23_09-53-09/quad1_model_2025-09-23_09-53-09/model_final.pth"  #adaptable-hound-205
DATA_ROOT="/home/xpetrus/Desktop/DP/Datasets/TNT_data/evaluations_datasets"
OUTPUT_FOLDER="./experiment-random-permutation-z-axis/"
DEVICE="cuda"
BATCH_SIZE=64
MODEL_DEPTH=5
MODEL="anisotropicunet"
HORIZONTAL_KERNEL="1,3,3"
HORIZONTAL_PADDING="0,1,1"

for QUAD in quad1 quad2 quad3 quad4
do
    echo "Evaluating $QUAD..."
    python evaluate_models.py \
        "${MODEL_PATH}" \
        "$DATA_ROOT/$QUAD" \
        "$OUTPUT_FOLDER" \
        --save_predictions \
        --device cuda \
        --batch_size ${BATCH_SIZE} \
        --model_type ${MODEL} \
        --model_depth ${MODEL_DEPTH} \
        --horizontal_kernel ${HORIZONTAL_KERNEL} \
        --horizontal_padding ${HORIZONTAL_PADDING}
done