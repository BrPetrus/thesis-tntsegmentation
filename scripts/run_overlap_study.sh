#!/bin/bash

# Automated overlap study for diploma thesis
# Evaluates all model runs in a directory with multiple tile overlaps

set -e

# Hardcoded configuration
OVERLAPS=(0 10 20 30 40)
EVAL_ROOT="/home/xpetrus/DP/Datasets/TNT_data/evaluations_datasets/"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# Get runs directory from argument
RUNS_DIR="${1:-}"

if [ -z "${RUNS_DIR}" ]; then
    echo "Usage: $0 <runs_directory>"
    echo ""
    echo "Example: $0 ./output/output-train-all-quads/"
    echo ""
    echo "The script will find all subdirectories containing trained models and"
    echo "evaluate each with overlaps: ${OVERLAPS[*]}"
    exit 1
fi

if [ ! -d "${RUNS_DIR}" ]; then
    echo "Error: Directory '${RUNS_DIR}' does not exist"
    exit 1
fi

# Create output directory for this study
OUTPUT_BASE="./output/overlap_study_${TIMESTAMP}"
mkdir -p "${OUTPUT_BASE}"

echo "==================================="
echo "OVERLAP STUDY (MULTI-RUN)"
echo "==================================="
echo "Runs directory: ${RUNS_DIR}"
echo "Output directory: ${OUTPUT_BASE}"
echo "Overlaps to test: ${OVERLAPS[*]}"
echo "==================================="
echo ""

# Find all run directories (directories containing *_model subdirectories)
echo "Scanning for run directories..."
RUN_DIRS=()

for dir in "${RUNS_DIR}"/*/ ; do
    if [ -d "${dir}" ]; then
        # Check if this directory has model subdirectories
        if ls "${dir}"/*_model/model_final.pth 1> /dev/null 2>&1; then
            RUN_DIRS+=("${dir}")
            echo "  Found: $(basename ${dir})"
        fi
    fi
done

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "Error: No run directories with trained models found in ${RUNS_DIR}"
    echo "Expected structure: ${RUNS_DIR}/*/quad*_model/model_final.pth"
    exit 1
fi

echo ""
echo "Found ${#RUN_DIRS[@]} run directories"
echo "Total evaluations: $((${#RUN_DIRS[@]} * ${#OVERLAPS[@]}))"
echo ""

# Initialize consolidated results CSV
CONSOLIDATED_CSV="${OUTPUT_BASE}/consolidated_results.csv"
echo "Run_Name,Overlap,Database_Path,Run_Name_Internal,Run_ID,Model_Signature,Train_Dice,Train_Jaccard,Eval_Dice_Mean,Eval_Dice_Std,Eval_Jaccard_Mean,Eval_Jaccard_Std,Eval_Accuracy_Mean,Eval_Accuracy_Std,Eval_Precision_Mean,Eval_Precision_Std,Eval_Recall_Mean,Eval_Recall_Std,Eval_Tversky_Mean,Eval_Tversky_Std,Eval_Focal_Tversky_Mean,Eval_Focal_Tversky_Std,Postprocess_Overall_Dice,Postprocess_Overall_Jaccard,Postprocess_Overall_Precision,Postprocess_Overall_Recall,Postprocess_Matched_Dice,Postprocess_Matched_Jaccard,Postprocess_Matched_Precision,Postprocess_Matched_Recall,Postprocess_Clean_Matched_Dice,Postprocess_Clean_Matched_Jaccard,Postprocess_Clean_Matched_Precision,Postprocess_Clean_Matched_Recall,Tunnel_TP,Tunnel_FP,Tunnel_FN,Tunnel_Precision,Tunnel_Recall,Tunnel_Dice,Tunnel_Jaccard,Unmatched_Predictions,Unmatched_Labels,Train_Quad,Test_Quad" > "${CONSOLIDATED_CSV}"

# Counter for progress
TOTAL_RUNS=$((${#RUN_DIRS[@]} * ${#OVERLAPS[@]}))
CURRENT_RUN=0

# Process each run directory
for run_dir in "${RUN_DIRS[@]}"; do
    RUN_NAME=$(basename "${run_dir}")
    
    echo "######################################"
    echo "# Processing: ${RUN_NAME}"
    echo "######################################"
    echo ""
    
    # Run evaluation for each overlap
    for overlap in "${OVERLAPS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        echo ""
        echo "[$CURRENT_RUN/$TOTAL_RUNS] ${RUN_NAME} with overlap=${overlap}px..."
        echo "-----------------------------------"
        
        # Create output folder for this run and overlap
        RUN_OUTPUT="${OUTPUT_BASE}/${RUN_NAME}_overlap_${overlap}"
        mkdir -p "${RUN_OUTPUT}"
        
        # Temporarily override OUTPUT_BASE in run.sh by modifying its behavior
        # We'll capture the output and move it
        TEMP_OUTPUT_BASE="./output/output-train-all-quads"
        
        # Run the evaluation using run.sh
        ./run.sh \
            --mode eval \
            --model-dir "${run_dir}" \
            --overlap_px ${overlap} \
            --run-postprocessing \
            --prediction-threshold 0.5 \
            --recall-threshold 0.5 \
            --minimum-size 100
        
        # Find the most recent output directory created by run.sh
        LATEST_RUN=$(ls -td ${TEMP_OUTPUT_BASE}/*/ 2>/dev/null | head -1)
        if [ -d "${LATEST_RUN}" ]; then
            echo "Moving results from ${LATEST_RUN} to ${RUN_OUTPUT}/"
            
            # Copy results file if it exists
            if [ -f "${LATEST_RUN}/all_results.csv" ]; then
                # Append to consolidated CSV with run name and overlap
                tail -n +2 "${LATEST_RUN}/all_results.csv" | while IFS=, read -r line; do
                    echo "${RUN_NAME},${overlap},${line}" >> "${CONSOLIDATED_CSV}"
                done
            fi
            
            # Move all files
            mv "${LATEST_RUN}"/* "${RUN_OUTPUT}/" 2>/dev/null || true
            rmdir "${LATEST_RUN}" 2>/dev/null || true
        fi
        
        echo "✓ Completed ${RUN_NAME} with overlap=${overlap}px"
    done
    
    echo ""
done

echo ""
echo "==================================="
echo "OVERLAP STUDY COMPLETE"
echo "==================================="
echo "Processed ${#RUN_DIRS[@]} run directories"
echo "Completed ${CURRENT_RUN} evaluations"
echo "Results saved to: ${OUTPUT_BASE}"
echo "Consolidated CSV: ${CONSOLIDATED_CSV}"
echo ""

# # Generate summary statistics
# if command -v python3 &> /dev/null && [ -f "${CONSOLIDATED_CSV}" ]; then
#     echo "Generating summary..."
#     python3 - <<EOF
# import pandas as pd
# import sys

# try:
#     df = pd.read_csv("${CONSOLIDATED_CSV}")
    
#     print("\n=== SUMMARY BY OVERLAP (ALL RUNS) ===")
#     summary = df.groupby('Overlap')[['Eval_Dice_Mean', 'Eval_Jaccard_Mean', 'Postprocess_Overall_Dice']].agg(['mean', 'std', 'min', 'max'])
#     print(summary)
    
#     print("\n=== SUMMARY BY RUN ===")
#     run_summary = df.groupby('Run_Name')[['Eval_Dice_Mean', 'Eval_Jaccard_Mean']].agg(['mean', 'std', 'count'])
#     print(run_summary)
    
#     print("\n=== BEST OVERLAP PER RUN-TRAIN-TEST COMBINATION ===")
#     best = df.loc[df.groupby(['Run_Name', 'Train_Quad', 'Test_Quad'])['Eval_Dice_Mean'].idxmax()]
#     print(best[['Run_Name', 'Train_Quad', 'Test_Quad', 'Overlap', 'Eval_Dice_Mean', 'Postprocess_Overall_Dice']].to_string(index=False))
    
#     print("\n=== TOP 10 OVERALL CONFIGURATIONS ===")
#     print(df.nlargest(10, 'Eval_Dice_Mean')[['Run_Name', 'Train_Quad', 'Test_Quad', 'Overlap', 'Eval_Dice_Mean', 'Postprocess_Overall_Dice']].to_string(index=False))
    
#     # Export summary to CSV
#     summary_file = "${OUTPUT_BASE}/overlap_summary.csv"
#     overlap_summary = df.groupby(['Run_Name', 'Overlap'])[['Eval_Dice_Mean', 'Eval_Jaccard_Mean', 'Postprocess_Overall_Dice']].mean()
#     overlap_summary.to_csv(summary_file)
#     print(f"\nSummary exported to: {summary_file}")
    
# except Exception as e:
#     print(f"Could not generate summary: {e}", file=sys.stderr)
#     import traceback
#     traceback.print_exc()
# EOF
# fi

# echo ""
echo "Done!"