#!/bin/bash
# This is a script used for training the model on all quads and evaluating it.

# NOTE: I am setting the ulimits here, but I still advise the user to change the niceness property of this script.
TIME=$((10*3600))
echo "Setting time limit to $(($TIME / 3600)) hours"
ulimit -t $TIME
export CUDA_VISIBLE_DEVICES=3
echo "Currently visible CUDA capable devices: ${CUDA_VISIBLE_DEVICES}"

# CLI option
MODE="both"  # Options: train, eval, both

# Training options
INPUT_ROOT="/home/xpetrus/Desktop/DP/Datasets/TNT_data/annotations/2025-10-01"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_BASE="./output/output-train-all-quads/${TIMESTAMP}"
EPOCHS=1000
WORKERS=4
BATCH=32
MLFLOW_SERVER_PORT=8800
LR=0.0001
MODEL=anisotropicunet_se
SEED=42
MODEL_DEPTH=5
WEIGHT_DECAY=0.0001
HORIZONTAL_KERNEL="1,3,3"
HORIZONTAL_PADDING="0,1,1"
QUADS=(quad1 quad2 quad3 quad4)

# Evaluation specific options
TILE_OVERLAP=20
MODEL_DIR=""  # Must be provided for eval-only mode
EVAL_ROOT="/home/xpetrus/DP/Datasets/TNT_data/evaluations_datasets/"

# Misc. options
RESULTS_CSV="${OUTPUT_BASE}/all_results.csv"
ENV_VAR="${OUTPUT_BASE}/env.txt"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --overlap_px)
            TILE_OVERLAP="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --mode [train|eval|both]     What to run (default: both)"
            echo "  --model-dir PATH             Directory containing trained models (required for eval mode)"
            echo "  --overlap_px N               Tile overlap in pixels (default: 20)"
            echo "  --help                       Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --mode train                                    # Train models only"
            echo "  $0 --mode eval --model-dir ./output/some-run       # Evaluate existing models"
            echo "  $0 --mode both                                     # Train then evaluate"
            echo "  $0 --mode eval --model-dir ./models --overlap_px 40 # Evaluate with custom overlap"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ "${MODE}" = "eval" ] && [ -z "${MODEL_DIR}" ]; then
    echo "Error: --model-dir is required when using --mode eval"
    echo "Use --help for more information"
    exit 1
fi

if [ "${MODE}" = "eval" ] && [ ! -d "${MODEL_DIR}" ]; then
    echo "Error: Model directory '${MODEL_DIR}' does not exist"
    exit 1
fi

# Initialize output directory and log mode
mkdir -p "${OUTPUT_BASE}"
echo "Running in mode: ${MODE}" | tee "${OUTPUT_BASE}/run_mode.log"

# Training function
run_training() {
    echo "=== Starting training phase ===" | tee -a "${OUTPUT_BASE}/run_mode.log"
    
    for i in "${!QUADS[@]}"; do
        QUAD=${QUADS[$i]}
        INPUT_FOLDER="${INPUT_ROOT}/${QUAD}"
        OUTPUT_FOLDER="${OUTPUT_BASE}/${QUAD}_model"

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
            return 1
        else
            echo "Successfully trained model for ${QUAD}."
        fi

        echo ""
    done
    return 0
}

# Evaluation function
run_evaluation() {
    echo "=== Starting evaluation phase ===" | tee -a "${OUTPUT_BASE}/run_mode.log"
    
    # Determine model base directory
    if [ "${MODE}" = "eval" ]; then
        MODEL_BASE_DIR="${MODEL_DIR}"
        echo "Using provided model directory: ${MODEL_BASE_DIR}"
    else
        MODEL_BASE_DIR="${OUTPUT_BASE}"
        echo "Using current run model directory: ${MODEL_BASE_DIR}"
    fi
    
    # Initialize CSV if it doesn't exist
    if [ ! -f "${RESULTS_CSV}" ]; then
        echo "Model,Train Quad,Test Quad,Tile_Overlap,Jaccard,Dice,Accuracy,Precision,Recall,Tversky,Focal Tversky" > "${RESULTS_CSV}"
    fi
    
    for i in "${!QUADS[@]}"; do
        TRAIN_QUAD=${QUADS[$i]}
        MODEL_PATH="${MODEL_BASE_DIR}/${TRAIN_QUAD}_model/model_final.pth"
        
        # Check if model exists
        if [ ! -f "${MODEL_PATH}" ]; then
            echo "Model for ${TRAIN_QUAD} not found at ${MODEL_PATH}. Skipping evaluation."
            continue
        fi
        
        echo "Evaluating model trained on ${TRAIN_QUAD} (${MODEL_PATH})..."
        
        # Evaluate on all quadrants
        for j in "${!QUADS[@]}"; do
            TEST_QUAD=${QUADS[$j]}
            TEST_DATA="${EVAL_ROOT}/${TEST_QUAD}"
            EVAL_OUTPUT="${OUTPUT_BASE}/${TRAIN_QUAD}_on_${TEST_QUAD}_overlap_${TILE_OVERLAP}"
            
            echo "  Testing on ${TEST_QUAD} with overlap ${TILE_OVERLAP}px..."
            mkdir -p "${EVAL_OUTPUT}"
            
            # Run evaluation script
            python evaluate_models.py \
                "${MODEL_PATH}" \
                "${TEST_DATA}" \
                "${EVAL_OUTPUT}" \
                --save_predictions \
                --device cuda \
                --batch_size ${BATCH} \
                --model_type ${MODEL} \
                --model_depth ${MODEL_DEPTH} \
                --horizontal_kernel ${HORIZONTAL_KERNEL} \
                --horizontal_padding ${HORIZONTAL_PADDING} \
                --tile_overlap ${TILE_OVERLAP}
            
            # Check for evaluation success
            if [ $? -ne 0 ]; then
                echo "Error evaluating ${TRAIN_QUAD} model on ${TEST_QUAD}. Continuing with next evaluation."
                continue
            fi 

            # Parse results and add to CSV
            METRICS_CSV="${EVAL_OUTPUT}/evaluation_metrics.csv"
            if [ -f "${METRICS_CSV}" ]; then
                # Try to find the line for the tested database path; fallback to last line
                LINE=$(grep -F "${TEST_DATA};" "${METRICS_CSV}" || true)
                if [ -z "${LINE}" ]; then
                    LINE=$(tail -n 1 "${METRICS_CSV}")
                fi

                # Parse semicolon-separated fields:
                IFS=';' read -r DBPATH DICE_MEAN DICE_STD JACCARD_MEAN JACCARD_STD ACCURACY_MEAN ACCURACY_STD PRECISION_MEAN PRECISION_STD RECALL_MEAN RECALL_STD TVERSKY_MEAN TVERSKY_STD FOCAL_TVERSKY_MEAN FOCAL_TVERSKY_STD <<< "${LINE}"
                echo "${MODEL},${TRAIN_QUAD},${TEST_QUAD},${TILE_OVERLAP},${JACCARD_MEAN},${DICE_MEAN},${ACCURACY_MEAN},${PRECISION_MEAN},${RECALL_MEAN},${TVERSKY_MEAN},${FOCAL_TVERSKY_MEAN}" >> "${RESULTS_CSV}"
            else
                echo "No metrics CSV found at ${METRICS_CSV}. Skipping results for this evaluation."
            fi
        done
    done
}

# Main execution based on mode
case "${MODE}" in
    "train")
        echo "Running training only..."
        run_training
        if [ $? -eq 0 ]; then
            echo "Training completed successfully!"
        else
            echo "Training failed!"
            exit 1
        fi
        ;;
    "eval")
        echo "Running evaluation only..."
        run_evaluation
        echo "Evaluation completed!"
        echo "Results saved to: ${RESULTS_CSV}"
        ;;
    "both")
        echo "Running both training and evaluation..."
        run_training
        if [ $? -eq 0 ]; then
            echo "Training completed successfully, starting evaluation..."
            run_evaluation
            echo "All training and evaluation completed!"
            echo "Results saved to: ${RESULTS_CSV}"
        else
            echo "Training failed, skipping evaluation!"
            exit 1
        fi
        ;;
    *)
        echo "Error: Unknown mode '${MODE}'. Use 'train', 'eval', or 'both'."
        exit 1
        ;;
esac

# Save environment variables
export -p >> "$ENV_VAR"

# Print summary
echo ""
echo "=== EXECUTION SUMMARY ==="
echo "Mode: ${MODE}"
echo "Timestamp: ${TIMESTAMP}"
echo "Output directory: ${OUTPUT_BASE}"
if [ "${MODE}" = "eval" ] || [ "${MODE}" = "both" ]; then
    echo "Tile overlap: ${TILE_OVERLAP}px"
    echo "Results CSV: ${RESULTS_CSV}"
fi
echo "Environment saved to: ${ENV_VAR}"