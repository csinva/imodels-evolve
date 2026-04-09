#!/bin/bash
# Setup Azure OpenAI Codex CLI with Entra ID Authentication
#
# Prerequisites:
#   1. Azure CLI installed and logged in: az login
#   2. azure-identity Python package: pip install azure-identity
#   3. "Cognitive Services OpenAI User" role on your Azure OpenAI resource
#   4. Codex CLI installed: npm install -g @openai/codex
#
# Usage:
#   source setup_azure.sh
#
# Environment variables (set before sourcing, or edit defaults below):
#   AZURE_RESOURCE_NAME   - your Azure OpenAI resource name
#   AZURE_DEPLOYMENT_NAME - your Azure deployment name (not model name)

set -euo pipefail

AZURE_RESOURCE_NAME="${AZURE_RESOURCE_NAME:-dl-openai-1}"
AZURE_DEPLOYMENT_NAME="${AZURE_DEPLOYMENT_NAME:-o4-mini}"
AZURE_API_VERSION="${AZURE_API_VERSION:-2025-03-01-preview}"
AZURE_WIRE_API="${AZURE_WIRE_API:-responses}"

# Create Codex CLI config
CODEX_CONFIG_DIR="$HOME/.codex"
mkdir -p "$CODEX_CONFIG_DIR"

cat > "$CODEX_CONFIG_DIR/config.toml" <<EOF
model_provider = "azure"
model = "${AZURE_DEPLOYMENT_NAME}"

[model_providers.azure]
name = "Azure OpenAI"
base_url = "https://${AZURE_RESOURCE_NAME}.openai.azure.com/openai"
query_params = { api-version = "${AZURE_API_VERSION}" }
wire_api = "${AZURE_WIRE_API}"
env_key = "AZURE_OPENAI_API_KEY"

[profiles.azure]
model_provider = "azure"
model = "${AZURE_DEPLOYMENT_NAME}"
EOF

echo "Wrote Codex config to $CODEX_CONFIG_DIR/config.toml"

# Get Entra ID token
export AZURE_OPENAI_API_KEY="$(python3 -c '
from azure.identity import AzureCliCredential
cred = AzureCliCredential()
print(cred.get_token("https://cognitiveservices.azure.com/.default").token)
')"

echo "Azure token exported to AZURE_OPENAI_API_KEY"
echo "Test with: npx codex --profile azure 'Say hello from Azure'"
