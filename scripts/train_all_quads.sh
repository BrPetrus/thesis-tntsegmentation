#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
echo "Currently visible CUDA capable devices: ${CUDA_VISIBLE_DEVICES}"

TIME=$((10*3600))
echo "Setting time limit to $(($TIME / 3600)) hours"
ulimit -t $TIME

INPUT_ROOT="/home/xpetrus/Desktop/DP/Datasets/TNT_data/annotations/2025-09-19"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_BASE="./output/output-train-all-quads/"
EPOCHS=10
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
RESULTS_CSV="${OUTPUT_BASE}/all_results_${TIMESTAMP}.csv"

# Initialize csvs
mkdir -p "${OUTPUT_BASE}"
echo "Model,Train Quad,Test Quad,TP,FP,FN,TN,Jaccard,Dice,Accuracy,Precision,Recall,Tversky,Focal Tversky" > "${RESULTS_CSV}"

# Training phase
echo "=== Starting training phase ==="
for i in "${!QUADS[@]}"; do
    QUAD=${QUADS[$i]}
    INPUT_FOLDER="${INPUT_FOLDER}/${QUAD}"
    OUTPUT_FOLDER="${OUTPUT_BASE}/${QUAD}_model_${TIMESTAMP}"

    echo "Training model for ${QUAD}..."
    echo "Input: ${QUAD_INPUT}"
    echo "Output: ${OUTPUT_FOLDER}/${QUAD}_model_${TIMESTAMP}"
    mkdir -p "${OUTPUT_FOLDER}"

    # Run training script
    python train.py \
        --input "${QUAD_INPUT}" \
        --output "${OUTPUT_FOLDER}" \
        --epochs ${EPOCHS} \
        --workers ${WORKERS} \
        --batch-size ${BATCH} \
        --mlflow-port ${MLFLOW_SERVER_PORT} \
        --lr ${LR} \
        --model ${MODEL} \
        --seed ${SEED} \
        --model-depth ${MODEL_DEPTH} \
        --weight-decay ${WEIGHT_DECAY} \
        --horizontal-kernel ${HORIZONTAL_KERNEL} \
        --horizontal-padding ${HORIZONTAL_PADDING}

    # Check for training success
    if [ $? -ne 0]; then
        echo "Error training model for ${QUAD}. Continuing with next quadrant."
    else
        echo "Successfully trained model for ${QUAD}."
    fi

    echo ""
done

# Evaluation phase
echo "=== Starting evaluation phase ==="
for i in "${!QUADS[@]}"; do
    TRAIN_QUAD=${QUADS[$i]}
    MODEL_PATH="${OUTPUT_BASE}/${TRAIN_QUAD}_model_${TIMESTAMP}/best_model.pth"
    
    # Check if model exists
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "Model for ${TRAIN_QUAD} not found at ${MODEL_PATH}. Skipping evaluation."
        continue
    fi
    
    echo "Evaluating model trained on ${TRAIN_QUAD}..."
    
    # Evaluate on all quadrants
    for j in "${!QUADS[@]}"; do
        TEST_QUAD=${QUADS[$j]}
        TEST_DATA="${INPUT_ROOT}/${TEST_QUAD}"
        EVAL_OUTPUT="${OUTPUT_BASE}/${TRAIN_QUAD}_on_${TEST_QUAD}_${TIMESTAMP}"
        
        echo "  Testing on ${TEST_QUAD}..."
        mkdir -p "${EVAL_OUTPUT}"
        
        # Run evaluation script
        python evaluate_models.py \
            --model-path "${MODEL_PATH}" \
            --data-path "${TEST_DATA}" \
            --output-dir "${EVAL_OUTPUT}" \
            --store-predictions \
            --model-type ${MODEL} \
            --model-depth ${MODEL_DEPTH} \
            --horizontal-kernel ${HORIZONTAL_KERNEL} \
            --horizontal-padding ${HORIZONTAL_PADDING}
        
        # Check for evaluation success
        if [ $? -ne 0 ]; then
            echo "Error evaluating ${TRAIN_QUAD} model on ${TEST_QUAD}. Continuing with next evaluation."
            continue
        fi 

        # Extract metrics from the output and append to CSV
        if [ -f "${EVAL_OUTPUT}/metrics.json" ]; then
            # Parse metrics.json and append to CSV
            # This is a simplified approach - you may need to adjust based on your actual output format
            TP=$(jq '.TP' "${EVAL_OUTPUT}/metrics.json")
            FP=$(jq '.FP' "${EVAL_OUTPUT}/metrics.json")
            FN=$(jq '.FN' "${EVAL_OUTPUT}/metrics.json")
            TN=$(jq '.TN' "${EVAL_OUTPUT}/metrics.json")
            JACCARD=$(jq '.jaccard' "${EVAL_OUTPUT}/metrics.json")
            DICE=$(jq '.dice' "${EVAL_OUTPUT}/metrics.json")
            ACCURACY=$(jq '.accuracy' "${EVAL_OUTPUT}/metrics.json")
            PRECISION=$(jq '.precision' "${EVAL_OUTPUT}/metrics.json")
            RECALL=$(jq '.recall' "${EVAL_OUTPUT}/metrics.json")
            TVERSKY=$(jq '.tversky' "${EVAL_OUTPUT}/metrics.json")
            FOCAL_TVERSKY=$(jq '.focal_tversky' "${EVAL_OUTPUT}/metrics.json")
            
            echo "${MODEL},${TRAIN_QUAD},${TEST_QUAD},${TP},${FP},${FN},${TN},${JACCARD},${DICE},${ACCURACY},${PRECISION},${RECALL},${TVERSKY},${FOCAL_TVERSKY}" >> "${RESULTS_CSV}"
        else
            echo "No metrics file found at ${EVAL_OUTPUT}/metrics.json. Skipping results for this evaluation."
        fi
    done
done

echo "All training and evaluation completed!"
echo "Results saved to: ${RESULT_CSV}"

# Create a summary of the results
echo "=== Summary of results ==="
echo "Average Dice scores by model:"
for i in "${!QUADS[@]}"; do
    TRAIN_QUAD=${QUADS[$i]}
    AVG_DICE=$(grep "${TRAIN_QUAD}" "${RESULTS_CSV}" | awk -F, '{sum+=$8; count++} END {print sum/count}')
    echo "Model trained on ${TRAIN_QUAD}: ${AVG_DICE}"
done