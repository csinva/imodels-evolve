#!/bin/bash
# Run OpenAI Codex on all 12 Blade datasets via Azure OpenAI.
#
# Prerequisites:
#   1. Run: source setup_azure.sh
#   2. Run: python prepare_run.py
#
# Usage:
#   bash run_all.sh                  # run all datasets
#   bash run_all.sh hurricane        # run single dataset
#   bash run_all.sh --skip-existing  # skip datasets with existing conclusion.txt

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

DATASETS=(
    affairs amtl boxes caschools crofoot fertility
    fish hurricane mortgage panda_nuts reading soccer teachingratings
)

SKIP_EXISTING=false
SINGLE_DATASET=""

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --skip-existing) SKIP_EXISTING=true ;;
        *) SINGLE_DATASET="$arg" ;;
    esac
done

if [[ -n "$SINGLE_DATASET" ]]; then
    DATASETS=("$SINGLE_DATASET")
fi

# Refresh Azure token
echo "Refreshing Azure token..."
source "$SCRIPT_DIR/refresh_token.sh"

run_dataset() {
    local dataset="$1"
    local run_dir="$OUTPUT_DIR/$dataset"

    if [[ ! -d "$run_dir" ]]; then
        echo "SKIP: $dataset (run directory not found, run prepare_run.py first)"
        return 1
    fi

    if [[ "$SKIP_EXISTING" == true && -f "$run_dir/conclusion.txt" ]]; then
        echo "SKIP: $dataset (conclusion.txt already exists)"
        return 0
    fi

    echo "============================================"
    echo "Running Codex on: $dataset"
    echo "============================================"

    cd "$run_dir"

    # Run Codex with full access (needed for bubblewrap-less environments)
    npx @openai/codex exec \
        --config model_reasoning_effort="high" \
        --sandbox danger-full-access \
        "Read the file AGENTS.md for instructions. Then write a python script called analysis.py and run it with python3 analysis.py. The script must create conclusion.txt. Do NOT just show code - you must actually write the file and execute it."

    if [[ -f "conclusion.txt" ]]; then
        echo "SUCCESS: $dataset - conclusion.txt written"
        cat conclusion.txt
    else
        echo "WARNING: $dataset - no conclusion.txt produced"
    fi

    echo ""
    cd "$SCRIPT_DIR"
}

# Track results
TOTAL=${#DATASETS[@]}
SUCCESS=0
FAILED=0

for dataset in "${DATASETS[@]}"; do
    # Refresh token periodically (tokens expire after ~1 hour)
    source "$SCRIPT_DIR/refresh_token.sh" 2>/dev/null || true

    if run_dataset "$dataset"; then
        ((SUCCESS++)) || true
    else
        ((FAILED++)) || true
    fi
done

echo "============================================"
echo "Complete: $SUCCESS/$TOTAL succeeded, $FAILED failed"
echo "============================================"
