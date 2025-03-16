#!/bin/bash

# This script encrypts the secrets.yml file using Ansible Vault

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

# Encrypt the secrets file
echo "Encrypting secrets file..."
ansible-vault encrypt "$SECRETS_FILE" --vault-password-file "$PASSWORD_FILE"

if [ $? -eq 0 ]; then
    echo "Secrets file encrypted successfully!"
else
    echo "Failed to encrypt secrets file."
    exit 1
fi

echo "Done!" 