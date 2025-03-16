#!/bin/bash

# This script decrypts the secrets.yml file using Ansible Vault

# Check if Ansible is installed
if ! command -v ansible-vault &> /dev/null; then
    echo "Ansible is not installed. Please install it first."
    exit 1
fi

# Path to the secrets file and password file
SECRETS_FILE="$(dirname "$0")/secrets.yml"
PASSWORD_FILE="$(dirname "$0")/vault_password.txt"

# Check if the files exist
if [ ! -f "$SECRETS_FILE" ]; then
    echo "Secrets file not found at $SECRETS_FILE"
    exit 1
fi

if [ ! -f "$PASSWORD_FILE" ]; then
    echo "Vault password file not found at $PASSWORD_FILE"
    exit 1
fi

# Decrypt the secrets file
echo "Decrypting secrets file..."
ansible-vault decrypt "$SECRETS_FILE" --vault-password-file "$PASSWORD_FILE"

if [ $? -eq 0 ]; then
    echo "Secrets file decrypted successfully!"
else
    echo "Failed to decrypt secrets file."
    exit 1
fi

echo "Done!" 