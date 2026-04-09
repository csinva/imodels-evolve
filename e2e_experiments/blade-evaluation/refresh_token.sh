#!/bin/bash
# Refresh Azure Entra ID token (tokens expire after ~1 hour).
# Usage: source refresh_token.sh

export AZURE_OPENAI_API_KEY="$(python3 -c '
from azure.identity import AzureCliCredential
cred = AzureCliCredential()
print(cred.get_token("https://cognitiveservices.azure.com/.default").token)
')"

echo "Azure token refreshed."
