#!/usr/bin/env bash
# run_loop.sh — Autonomous research loop for Codex CLI
# Repeatedly invokes Codex in non-interactive mode to run one experiment per iteration.
#
# Usage:
#   ./run_loop.sh              # run with defaults
#   MAX_ITERATIONS=10 ./run_loop.sh  # override iteration count

set -uo pipefail
# Note: not using set -e because Codex exec may exit non-zero and we want to continue looping

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"  # seconds between iterations

# Azure Entra ID token refresh (keyless auth)
refresh_token() {
    export AZURE_OPENAI_API_KEY="$(uv run python3 -c '
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential
cred = ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential())
print(cred.get_token("https://cognitiveservices.azure.com/.default").token)
')"
    echo "Azure token refreshed (length: ${#AZURE_OPENAI_API_KEY})" >&2
}

refresh_token

# Create branch if not already on an autoresearch branch
CURRENT_BRANCH="$(git branch --show-current)"
if [[ ! "$CURRENT_BRANCH" =~ ^autoresearch/ ]]; then
    TAG="$(date +%b%d | tr '[:upper:]' '[:lower:]')"
    BRANCH="autoresearch/${TAG}-codex"
    echo "Creating branch: $BRANCH"
    git checkout -b "$BRANCH"
else
    echo "Already on branch: $CURRENT_BRANCH"
fi

# Write the prompt to a temp file (avoids stdin pipe issues with Codex)
PROMPT_FILE="$(mktemp)"
cat program.md > "$PROMPT_FILE"
cat >> "$PROMPT_FILE" <<'ENDPROMPT'

You are continuing an autonomous research loop. Read results/overall_results.csv to see what has been tried.
Run ONE experiment iteration: pick a new idea, edit interpretable_regressor.py, commit, evaluate, and record results.
Read through the evaluation in the `src` file to better understand how to improve results.
Be creative and try something different from what's already been tried.
ENDPROMPT
trap 'rm -f "$PROMPT_FILE"' EXIT

echo "=== Starting Codex autoresearch loop (max $MAX_ITERATIONS iterations) ==="
echo ""

for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Iteration $i / $MAX_ITERATIONS"
    echo "  $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Refresh Azure token (tokens expire after ~1 hour)
    refresh_token

    # Run Codex in non-interactive exec mode
    # Using --dangerously-bypass-approvals-and-sandbox because bubblewrap sandbox
    # doesn't work in this environment (loopback namespace not permitted).
    # The environment is already externally sandboxed.
    npx @openai/codex exec \
        --dangerously-bypass-approvals-and-sandbox \
        -C "$SCRIPT_DIR" \
        "$(cat "$PROMPT_FILE")" \
        < /dev/null \
        2>&1 | tee "logs/iteration_${i}.log" || {
            echo "WARNING: Codex iteration $i exited with error, continuing..."
        }

    echo ""
    echo "--- Results after iteration $i ---"
    tail -5 results/overall_results.csv 2>/dev/null || echo "(no results yet)"
    echo ""

    if [ "$i" -lt "$MAX_ITERATIONS" ]; then
        sleep "$SLEEP_BETWEEN"
    fi
done

echo ""
echo "=== Autoresearch loop complete ($MAX_ITERATIONS iterations) ==="
echo "Final results:"
cat results/overall_results.csv
