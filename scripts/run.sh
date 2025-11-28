#!/bin/bash
# This is a script used for training the model on all quads and evaluating it.

# NOTE: I am setting the ulimits here, but I still advise the user to change the niceness property of this script.
TIME=$((10*3600))
echo "Setting time limit to $(($TIME / 3600)) hours"
ulimit -t $TIME
echo "Currently visible CUDA capable devices: ${CUDA_VISIBLE_DEVICES}"

# CLI option
MODE="both"  # Options: train, eval, both

# Training options
INPUT_ROOT="/home/xpetrus/Desktop/DP/Datasets/TNT_data/annotations/2025-10-01"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_BASE=""  # Will be set via CLI or default
DEFAULT_OUTPUT_BASE="./output/output-train-all-quads/${TIMESTAMP}"
EPOCHS=1000
WORKERS=4
BATCH=32
MLFLOW_SERVER_PORT=8800
LR=0.0001
MODEL=anisotropicunet_usenet
SEED=42
MODEL_DEPTH=3
WEIGHT_DECAY=0.0001
HORIZONTAL_KERNEL="1,3,3"
HORIZONTAL_PADDING="0,1,1"
DOWNSCALE_KERNEL="1,2,2"
DOWNSCALE_STRIDE="1,2,2"
UPSCALE_STRIDE="1,2,2"
UPSCALE_KERNEL="1,2,2"
QUADS=(quad1 quad2 quad3 quad4)

# Evaluation specific options
TILE_OVERLAP=20
MODEL_DIR="/home/xpetrus/DP/DP-WIP/scripts/output/output-train-all-quads/2025-10-28_20-02-21/"  # Must be provided for eval-only mode
EVAL_ROOT="/home/xpetrus/DP/Datasets/TNT_data/evaluations_datasets/"
EVAL_SAME_QUAD_ONLY=false

# Postprocessing options
RUN_POSTPROCESSING=true
PREDICTION_THRESHOLD=0.5
RECALL_THRESHOLD=0.5
MINIMUM_SIZE=100

# Misc. options - will be updated after OUTPUT_BASE is set
RESULTS_CSV=""
ENV_VAR=""

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
        --output-dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --overlap_px)
            TILE_OVERLAP="$2"
            shift 2
            ;;
        --run-postprocessing)
            RUN_POSTPROCESSING=true
            shift
            ;;
        --eval-same-quad-only)
            EVAL_SAME_QUAD_ONLY=true
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --prediction-threshold)
            PREDICTION_THRESHOLD="$2"
            shift 2
            ;;
        --recall-threshold)
            RECALL_THRESHOLD="$2"
            shift 2
            ;;
        --minimum-size)
            MINIMUM_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --mode [train|eval|both]     What to run (default: both)"
            echo "  --model-dir PATH             Directory containing trained models (required for eval mode)"
            echo "  --output-dir PATH            Custom output directory (default: ./output/output-train-all-quads/TIMESTAMP)"
            echo "  --overlap_px N               Tile overlap in pixels (default: 0)"
            echo "  --seed N                     Random seed for training"
            echo "  --run-postprocessing         Enable tunnel-level postprocessing analysis"
            echo "  --eval-same-quad-only        Only evaluate each model on its corresponding quadrant"
            echo "  --prediction-threshold F     Threshold for binarizing predictions (default: 0.5)"
            echo "  --recall-threshold F         Recall threshold for tunnel matching (default: 0.5)"
            echo "  --minimum-size N             Minimum size for connected components (default: 100)"
            echo "  --help                       Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --mode train                                    # Train models only"
            echo "  $0 --mode eval --model-dir ./output/some-run       # Evaluate existing models"
            echo "  $0 --mode both                                     # Train then evaluate"
            echo "  $0 --mode eval --model-dir ./models --overlap_px 40 # Evaluate with custom overlap"
            echo "  $0 --mode eval --model-dir ./models --run-postprocessing # Evaluate with postprocessing"
            echo "  $0 --mode eval --model-dir ./models --eval-same-quad-only # Evaluate only matching quads"
            echo "  $0 --mode eval --model-dir ./models --output-dir /tmp/results # Custom output location"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set OUTPUT_BASE to default if not provided
if [ -z "${OUTPUT_BASE}" ]; then
    OUTPUT_BASE="${DEFAULT_OUTPUT_BASE}"
fi

# Set dependent paths
RESULTS_CSV="${OUTPUT_BASE}/all_results.csv"
ENV_VAR="${OUTPUT_BASE}/env.txt"

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
echo "Postprocessing enabled: ${RUN_POSTPROCESSING}" | tee -a "${OUTPUT_BASE}/run_mode.log"
echo "Eval same quad only: ${EVAL_SAME_QUAD_ONLY}" | tee -a "${OUTPUT_BASE}/run_mode.log"

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

        SEED_THIS=$((SEED + i))

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
            --seed ${SEED_THIS} \
            --model_depth ${MODEL_DEPTH} \
            --weight_decay ${WEIGHT_DECAY} \
            --horizontal_kernel ${HORIZONTAL_KERNEL} \
            --horizontal_padding ${HORIZONTAL_PADDING} \
            --downscale_kernel ${DOWNSCALE_KERNEL} \
            --downscale_stride ${DOWNSCALE_STRIDE} \
            --upscale_kernel ${UPSCALE_KERNEL} \
            --upscale_stride ${UPSCALE_STRIDE} \
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
        echo "Database_Path,Run_Name,Run_ID,Model_Signature,Train_Dice,Train_Jaccard,Eval_Dice_Mean,Eval_Dice_Std,Eval_Jaccard_Mean,Eval_Jaccard_Std,Eval_Accuracy_Mean,Eval_Accuracy_Std,Eval_Precision_Mean,Eval_Precision_Std,Eval_Recall_Mean,Eval_Recall_Std,Eval_Tversky_Mean,Eval_Tversky_Std,Eval_Focal_Tversky_Mean,Eval_Focal_Tversky_Std,Postprocess_Overall_Dice,Postprocess_Overall_Jaccard,Postprocess_Overall_Precision,Postprocess_Overall_Recall,Postprocess_Matched_Dice,Postprocess_Matched_Jaccard,Postprocess_Matched_Precision,Postprocess_Matched_Recall,Postprocess_Clean_Matched_Dice,Postprocess_Clean_Matched_Jaccard,Postprocess_Clean_Matched_Precision,Postprocess_Clean_Matched_Recall,Tunnel_TP,Tunnel_FP,Tunnel_FN,Tunnel_Precision,Tunnel_Recall,Tunnel_Dice,Tunnel_Jaccard,Unmatched_Predictions,Unmatched_Labels,Train_Quad,Test_Quad,Tile_Overlap" > "${RESULTS_CSV}"
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
        
        # Determine which quadrants to evaluate on
        if [ "${EVAL_SAME_QUAD_ONLY}" = true ]; then
            # Only evaluate on the same quadrant
            TEST_QUADS=("${TRAIN_QUAD}")
            echo "  Evaluating only on matching quadrant: ${TRAIN_QUAD}"
        else
            # Evaluate on all quadrants
            TEST_QUADS=("${QUADS[@]}")
            echo "  Evaluating on all quadrants"
        fi
        
        # Evaluate on selected quadrants
        for TEST_QUAD in "${TEST_QUADS[@]}"; do
            TEST_DATA="${EVAL_ROOT}/${TEST_QUAD}"
            EVAL_OUTPUT="${OUTPUT_BASE}/${TRAIN_QUAD}_on_${TEST_QUAD}_overlap_${TILE_OVERLAP}"
            
            echo "  Testing on ${TEST_QUAD} with overlap ${TILE_OVERLAP}px..."
            mkdir -p "${EVAL_OUTPUT}"
            
            # Build evaluation command
            EVAL_CMD="python evaluate_models.py \
                \"${MODEL_PATH}\" \
                \"${TEST_DATA}\" \
                \"${EVAL_OUTPUT}\" \
                --save_predictions \
                --device cuda \
                --batch_size ${BATCH} \
                --tile_overlap ${TILE_OVERLAP}"
            
            # Add postprocessing arguments if enabled
            if [ "${RUN_POSTPROCESSING}" = true ]; then
                EVAL_CMD="${EVAL_CMD} \
                    --run_postprocessing \
                    --prediction_threshold ${PREDICTION_THRESHOLD} \
                    --recall_threshold ${RECALL_THRESHOLD} \
                    --minimum_size ${MINIMUM_SIZE}"
                echo "    Postprocessing enabled (pred_thresh=${PREDICTION_THRESHOLD}, recall_thresh=${RECALL_THRESHOLD})"
            fi
            
            # Run evaluation script
            eval ${EVAL_CMD}
            
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

                # Parse semicolon-separated fields
                IFS=';' read -r DBPATH RUN_NAME RUN_ID MODEL_SIGNATURE TRAIN_DICE TRAIN_JACCARD EVAL_DICE_MEAN EVAL_DICE_STD EVAL_JACCARD_MEAN EVAL_JACCARD_STD EVAL_ACCURACY_MEAN EVAL_ACCURACY_STD EVAL_PRECISION_MEAN EVAL_PRECISION_STD EVAL_RECALL_MEAN EVAL_RECALL_STD EVAL_TVERSKY_MEAN EVAL_TVERSKY_STD EVAL_FOCAL_TVERSKY_MEAN EVAL_FOCAL_TVERSKY_STD POSTPROCESS_OVERALL_DICE POSTPROCESS_OVERALL_JACCARD POSTPROCESS_OVERALL_PRECISION POSTPROCESS_OVERALL_RECALL POSTPROCESS_MATCHED_DICE POSTPROCESS_MATCHED_JACCARD POSTPROCESS_MATCHED_PRECISION POSTPROCESS_MATCHED_RECALL POSTPROCESS_CLEAN_MATCHED_DICE POSTPROCESS_CLEAN_MATCHED_JACCARD POSTPROCESS_CLEAN_MATCHED_PRECISION POSTPROCESS_CLEAN_MATCHED_RECALL TUNNEL_TP TUNNEL_FP TUNNEL_FN TUNNEL_PRECISION TUNNEL_RECALL TUNNEL_DICE TUNNEL_JACCARD UNMATCHED_PREDICTIONS UNMATCHED_LABELS <<< "${LINE}"
                
                # Write to results CSV with all fields
                echo "${DBPATH},${RUN_NAME},${RUN_ID},${MODEL_SIGNATURE},${TRAIN_DICE},${TRAIN_JACCARD},${EVAL_DICE_MEAN},${EVAL_DICE_STD},${EVAL_JACCARD_MEAN},${EVAL_JACCARD_STD},${EVAL_ACCURACY_MEAN},${EVAL_ACCURACY_STD},${EVAL_PRECISION_MEAN},${EVAL_PRECISION_STD},${EVAL_RECALL_MEAN},${EVAL_RECALL_STD},${EVAL_TVERSKY_MEAN},${EVAL_TVERSKY_STD},${EVAL_FOCAL_TVERSKY_MEAN},${EVAL_FOCAL_TVERSKY_STD},${POSTPROCESS_OVERALL_DICE},${POSTPROCESS_OVERALL_JACCARD},${POSTPROCESS_OVERALL_PRECISION},${POSTPROCESS_OVERALL_RECALL},${POSTPROCESS_MATCHED_DICE},${POSTPROCESS_MATCHED_JACCARD},${POSTPROCESS_MATCHED_PRECISION},${POSTPROCESS_MATCHED_RECALL},${POSTPROCESS_CLEAN_MATCHED_DICE},${POSTPROCESS_CLEAN_MATCHED_JACCARD},${POSTPROCESS_CLEAN_MATCHED_PRECISION},${POSTPROCESS_CLEAN_MATCHED_RECALL},${TUNNEL_TP},${TUNNEL_FP},${TUNNEL_FN},${TUNNEL_PRECISION},${TUNNEL_RECALL},${TUNNEL_DICE},${TUNNEL_JACCARD},${UNMATCHED_PREDICTIONS},${UNMATCHED_LABELS},${TRAIN_QUAD},${TEST_QUAD},${TILE_OVERLAP}" >> "${RESULTS_CSV}"
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
    echo "Eval same quad only: ${EVAL_SAME_QUAD_ONLY}"
    echo "Postprocessing: ${RUN_POSTPROCESSING}"
    if [ "${RUN_POSTPROCESSING}" = true ]; then
        echo "  Minimum size: ${MINIMUM_SIZE}"
        echo "  Prediction threshold: ${PREDICTION_THRESHOLD}"
        echo "  Recall threshold: ${RECALL_THRESHOLD}"
    fi
    echo "Results CSV: ${RESULTS_CSV}"
fi
echo "Environment saved to: ${ENV_VAR}"
