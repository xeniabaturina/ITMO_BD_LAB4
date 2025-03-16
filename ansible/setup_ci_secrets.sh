#!/bin/bash

# This script sets up the secrets for the CI/CD pipeline

# Check if Ansible is installed
if ! command -v ansible-vault &> /dev/null; then
    echo "Error: Ansible is not installed. Please install it first."
    exit 1
fi

# Path to the secrets file
SECRETS_FILE="$(dirname "$0")/secrets.yml"
SECRETS_DIR="${SECRETS_DIR:-../secrets}"

# Check if the file exists
if [ ! -f "$SECRETS_FILE" ]; then
    echo "Error: Secrets file not found at $SECRETS_FILE"
    exit 1
fi

# Check if VAULT_PASSWORD environment variable is set
if [ -z "$VAULT_PASSWORD" ]; then
    echo "Error: VAULT_PASSWORD environment variable is not set."
    echo "Please set it with: export VAULT_PASSWORD=your-secure-password"
    exit 1
fi

# Create secrets directory if it doesn't exist
mkdir -p "$SECRETS_DIR"

# Create a temporary file for the vault password
TEMP_PASSWORD_FILE=$(mktemp)
echo "$VAULT_PASSWORD" > "$TEMP_PASSWORD_FILE"

# Decrypt the secrets file and extract the credentials
echo "Extracting secrets from vault..."
ansible-vault view "$SECRETS_FILE" --vault-password-file "$TEMP_PASSWORD_FILE" | \
python3 -c '
import yaml
import sys
import os

data = yaml.safe_load(sys.stdin)
creds = data["db_credentials"]
secrets_dir = os.environ.get("SECRETS_DIR", "../secrets")

for k, v in creds.items():
    with open(os.path.join(secrets_dir, k), "w") as f:
        f.write(str(v))
    print(f"Created secret file for {k}")
'

# Check if extraction was successful
if [ $? -eq 0 ]; then
    echo "Secrets extracted successfully!"
else
    echo "Failed to extract secrets."
    rm "$TEMP_PASSWORD_FILE"
    exit 1
fi

# Remove the temporary file
rm "$TEMP_PASSWORD_FILE"

echo "Done!"
