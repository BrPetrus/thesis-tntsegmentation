#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
echo "Currently visible CUDA capable devices: ${CUDA_VISIBLE_DEVICES}"

TIME=$((10*3600))
echo "Setting time limit to $(($TIME / 3600)) hours"
ulimit -t $TIME

INPUT_ROOT="/home/xpetrus/Desktop/DP/Datasets/TNT_data/annotations/2025-09-19"
EVAL_ROOT="/home/xpetrus/DP/Datasets/TNT_data/evaluations_datasets/"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_BASE="./output/output-train-all-quads/${TIMESTAMP}"
EPOCHS=1000
WORKERS=4
BATCH=32
MLFLOW_SERVER_PORT=8800
LR=0.0001
MODEL=anisotropicunet
SEED=42
MODEL_DEPTH=5
WEIGHT_DECAY=0.0001
HORIZONTAL_KERNEL="3,3,3"
HORIZONTAL_PADDING="1,1,1"
QUADS=(quad1 quad2 quad3 quad4)
RESULTS_CSV="${OUTPUT_BASE}/all_results_${TIMESTAMP}.csv"

# Initialize csvs
mkdir -p "${OUTPUT_BASE}"
echo "Model,Train Quad,Test Quad,Jaccard,Dice,Accuracy,Precision,Recall,Tversky,Focal Tversky" > "${RESULTS_CSV}"

# Training phase
echo "=== Starting training phase ==="
for i in "${!QUADS[@]}"; do
    QUAD=${QUADS[$i]}
    INPUT_FOLDER="${INPUT_ROOT}/${QUAD}"
    OUTPUT_FOLDER="${OUTPUT_BASE}/${QUAD}_model_${TIMESTAMP}"

    echo "Training model for ${QUAD}..."
    echo "Input: ${INPUT_FOLDER}"
    echo "Output: ${OUTPUT_FOLDER}"
    mkdir -p "${OUTPUT_FOLDER}"

    # Run training script
    python training.py \
        "${INPUT_FOLDER}" \
        "${OUTPUT_FOLDER}" \
        --epochs ${EPOCHS} \
        --num_workers ${WORKERS} \
        --batch_size ${BATCH} \
        --mlflow_port ${MLFLOW_SERVER_PORT} \
        --lr ${LR} \
        --model ${MODEL} \
        --seed ${SEED} \
        --model_depth ${MODEL_DEPTH} \
        --weight_decay ${WEIGHT_DECAY} \
        --horizontal_kernel ${HORIZONTAL_KERNEL} \
        --horizontal_padding ${HORIZONTAL_PADDING} \
        --shuffle

    # Check for training success
    if [ $? -ne 0 ]; then
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
    MODEL_PATH="${OUTPUT_BASE}/${TRAIN_QUAD}_model_${TIMESTAMP}/model_final.pth"
    
    # Check if model exists
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "Model for ${TRAIN_QUAD} not found at ${MODEL_PATH}. Skipping evaluation."
        continue
    fi
    
    echo "Evaluating model trained on ${TRAIN_QUAD}..."
    
    # Evaluate on all quadrants
    for j in "${!QUADS[@]}"; do
        TEST_QUAD=${QUADS[$j]}
        TEST_DATA="${EVAL_ROOT}/${TEST_QUAD}"
        EVAL_OUTPUT="${OUTPUT_BASE}/${TRAIN_QUAD}_on_${TEST_QUAD}_${TIMESTAMP}"
        
        echo "  Testing on ${TEST_QUAD}..."
        mkdir -p "${EVAL_OUTPUT}"
        
        # Run evaluation script
        python evaluate_models.py \
            "${MODEL_PATH}" \
            "${TEST_DATA}" \
            "${EVAL_OUTPUT}" \
            --save_predictions \
            --device cuda \
            --batch_size ${BATCH}
        
        # Check for evaluation success
        if [ $? -ne 0 ]; then
            echo "Error evaluating ${TRAIN_QUAD} model on ${TEST_QUAD}. Continuing with next evaluation."
            continue
        fi 

        # evaluate_models.py writes EVAL_OUTPUT/evaluation_metrics.csv with semicolon separators.
        METRICS_CSV="${EVAL_OUTPUT}/evaluation_metrics.csv"
        if [ -f "${METRICS_CSV}" ]; then
            # Try to find the line for the tested database path; fallback to last line
            LINE=$(grep -F "${TEST_DATA};" "${METRICS_CSV}" || true)
            if [ -z "${LINE}" ]; then
                LINE=$(tail -n 1 "${METRICS_CSV}")
            fi

            # Parse semicolon-separated fields:
            IFS=';' read -r DBPATH DICE_MEAN DICE_STD JACCARD_MEAN JACCARD_STD ACCURACY_MEAN ACCURACY_STD PRECISION_MEAN PRECISION_STD RECALL_MEAN RECALL_STD TVERSKY_MEAN TVVERSKY_STD FOCAL_TVERSKY_MEAN FOCAL_TVERSKY_STD <<< "${LINE}"
            echo "${MODEL},${TRAIN_QUAD},${TEST_QUAD},${JACCARD_MEAN},${DICE_MEAN},${ACCURACY_MEAN},${PRECISION_MEAN},${RECALL_MEAN},${TVERSKY_MEAN},${FOCAL_TVERSKY_MEAN}" >> "${RESULTS_CSV}"
        else
            echo "No metrics CSV found at ${METRICS_CSV}. Skipping results for this evaluation."
        fi
    done
done

echo "All training and evaluation completed!"
echo "Results saved to: ${RESULT_CSV}"
