#!/bin/bash
# Run Claude Code on all 13 Blade datasets (non-interactive via `claude -p`).
#
# Prerequisites:
#   1. Claude CLI authenticated (`claude auth` or ANTHROPIC_API_KEY)
#   2. python prepare_run.py --mode <standard|custom_v2>
#
# Usage:
#   bash run_all.sh --mode standard              # run all datasets with standard tools
#   bash run_all.sh --mode custom_v2             # run all datasets with custom tools
#   bash run_all.sh --mode standard hurricane    # run single dataset
#   bash run_all.sh --mode standard --skip-existing

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATASETS=(
    affairs amtl boxes caschools crofoot fertility
    fish hurricane mortgage panda_nuts reading soccer teachingratings
)

SKIP_EXISTING=false
SINGLE_DATASET=""
MODE="standard"
CUSTOM_OUTPUT_DIR=""
CLAUDE_MODEL="${CLAUDE_MODEL:-sonnet}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --mode) MODE="$2"; shift 2 ;;
        --output-dir) CUSTOM_OUTPUT_DIR="$2"; shift 2 ;;
        --model) CLAUDE_MODEL="$2"; shift 2 ;;
        *) SINGLE_DATASET="$1"; shift ;;
    esac
done

if [[ -n "$CUSTOM_OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/$CUSTOM_OUTPUT_DIR"
else
    OUTPUT_DIR="$SCRIPT_DIR/outputs_${MODE}"
fi

if [[ -n "$SINGLE_DATASET" ]]; then
    DATASETS=("$SINGLE_DATASET")
fi

echo "Mode: $MODE"
echo "Output dir: $OUTPUT_DIR"
echo "Claude model: $CLAUDE_MODEL"

run_dataset() {
    local dataset="$1"
    local run_dir="$OUTPUT_DIR/$dataset"

    if [[ ! -d "$run_dir" ]]; then
        echo "SKIP: $dataset (run directory not found, run prepare_run.py --mode $MODE first)"
        return 1
    fi

    if [[ "$SKIP_EXISTING" == true && -f "$run_dir/conclusion.txt" ]]; then
        echo "SKIP: $dataset (conclusion.txt already exists)"
        return 0
    fi

    # Remove stale output if re-running
    rm -f "$run_dir/conclusion.txt" "$run_dir/analysis.py"

    echo "============================================"
    echo "Running Claude on: $dataset (mode: $MODE)"
    echo "============================================"

    cd "$run_dir"

    # Run Claude Code non-interactively with full tool access in this dir.
    # Hard time-cap each run so a stuck agent does not block the whole pipeline
    # (Claude sometimes keeps its session open after writing conclusion.txt).
    local prompt='Read the file AGENTS.md for instructions. Then write a python script called analysis.py and run it with python3 analysis.py. The script must create conclusion.txt. Do NOT just show code - you must actually write the file and execute it. Once conclusion.txt exists, stop immediately.'
    printf '%s' "$prompt" | timeout --kill-after=30s 900s claude -p \
        --model "$CLAUDE_MODEL" \
        --permission-mode bypassPermissions \
        > claude_stdout.log 2> claude_stderr.log || true

    if [[ -f "conclusion.txt" ]]; then
        echo "SUCCESS: $dataset - conclusion.txt written"
        cat conclusion.txt
    else
        echo "WARNING: $dataset - no conclusion.txt produced"
        tail -20 claude_stderr.log 2>/dev/null || true
    fi

    echo ""
    cd "$SCRIPT_DIR"
}

# Track results
TOTAL=${#DATASETS[@]}
SUCCESS=0
FAILED=0

for dataset in "${DATASETS[@]}"; do
    if run_dataset "$dataset"; then
        ((SUCCESS++)) || true
    else
        ((FAILED++)) || true
    fi
done

echo "============================================"
echo "Complete ($MODE mode): $SUCCESS/$TOTAL succeeded, $FAILED failed"
echo "============================================"
