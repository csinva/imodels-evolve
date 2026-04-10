#!/bin/bash
# Run OpenAI Codex on all 13 Blade datasets via Azure OpenAI (keyless Entra ID auth).
#
# Prerequisites:
#   1. az login (Azure CLI)
#   2. python prepare_run.py
#   3. ~/.codex/config.toml pointing to dl-openai-3 / gpt-5.3-codex
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

# Get Azure Entra ID token (keyless)
refresh_token() {
    export AZURE_OPENAI_API_KEY="$(python3 -c '
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential
cred = ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential())
print(cred.get_token("https://cognitiveservices.azure.com/.default").token)
')"
    echo "Azure token refreshed (length: ${#AZURE_OPENAI_API_KEY})" >&2
}

echo "Refreshing Azure token..."
refresh_token

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

    # Remove stale conclusion if re-running
    rm -f "$run_dir/conclusion.txt" "$run_dir/analysis.py"

    echo "============================================"
    echo "Running Codex on: $dataset"
    echo "============================================"

    cd "$run_dir"

    # Run Codex (gpt-5.3-codex on dl-openai-3 via ~/.codex/config.toml)
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
    refresh_token 2>/dev/null || true

    if run_dataset "$dataset"; then
        ((SUCCESS++)) || true
    else
        ((FAILED++)) || true
    fi
done

echo "============================================"
echo "Complete: $SUCCESS/$TOTAL succeeded, $FAILED failed"
echo "============================================"
