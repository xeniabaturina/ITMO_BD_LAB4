#!/usr/bin/env python3
"""
Simplified test script to verify the Ansible Vault setup.
This script tests:
1. The secrets file is properly encrypted
2. The secrets can be decrypted and have the correct structure
3. The secrets_manager can read the secrets
"""

import os
import sys
import subprocess
import yaml
import json
from pathlib import Path

def run_command(command):
    """Run a shell command and return the output"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Error: {result.stderr}")
        return None
    return result.stdout.strip()

def is_file_encrypted(file_path):
    """Check if a file is encrypted with Ansible Vault"""
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        return first_line.startswith('$ANSIBLE_VAULT')

def test_vault_encryption():
    """Test that the vault is properly encrypted"""
    print("\n=== Testing Vault Encryption ===")
    
    # Get the path to the secrets file
    script_dir = Path(__file__).parent
    secrets_file = script_dir / "ansible" / "secrets.yml"
    
    # Check if the file exists
    if not secrets_file.exists():
        print(f"Error: Secrets file not found at {secrets_file}")
        return False
    
    # Check if the file is encrypted
    if not is_file_encrypted(secrets_file):
        print("Error: The secrets file is not encrypted")
        return False
    
    print("✅ Secrets file is properly encrypted")
    return True

def test_vault_decryption():
    """Test that the vault can be decrypted"""
    print("\n=== Testing Vault Decryption ===")
    
    # Get the path to the secrets file and password file
    script_dir = Path(__file__).parent
    secrets_file = script_dir / "ansible" / "secrets.yml"
    password_file = script_dir / "ansible" / "vault_password.txt"
    
    # Check if the files exist
    if not secrets_file.exists():
        print(f"Error: Secrets file not found at {secrets_file}")
        return False
    
    if not password_file.exists():
        print(f"Error: Vault password file not found at {password_file}")
        return False
    
    # View the secrets file
    view_result = run_command(f"ansible-vault view {secrets_file} --vault-password-file {password_file}")
    if view_result is None:
        print("Error: Failed to decrypt the secrets file")
        return False
    
    # Parse the YAML
    try:
        secrets = yaml.safe_load(view_result)
        
        # Check if the expected keys are present
        if 'db_credentials' not in secrets:
            print("Error: 'db_credentials' key not found in secrets")
            return False
        
        db_creds = secrets['db_credentials']
        required_keys = ['postgres_user', 'postgres_password', 'postgres_db', 'postgres_schema']
        
        for key in required_keys:
            if key not in db_creds:
                print(f"Error: '{key}' not found in db_credentials")
                return False
        
        # Print the credentials (masked for security)
        masked_creds = {k: v[:2] + '*****' if k.endswith('password') else v for k, v in db_creds.items()}
        print(f"Database credentials: {json.dumps(masked_creds, indent=2)}")
        
        print("✅ Secrets file can be decrypted and has the correct structure")
        return True
    
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return False

def test_secrets_manager():
    """Test that the secrets_manager can read the secrets"""
    print("\n=== Testing Secrets Manager ===")
    
    try:
        # Create a temporary environment for testing
        script_dir = Path(__file__).parent
        os.environ['VAULT_FILE_PATH'] = str(script_dir / "ansible" / "secrets.yml")
        os.environ['VAULT_PASSWORD_FILE'] = str(script_dir / "ansible" / "vault_password.txt")
        
        # Add the current directory to the path
        sys.path.insert(0, str(script_dir))
        
        # Import the secrets_manager
        from src.secrets_manager import get_secrets_manager
        
        # Get the secrets manager
        secrets_manager = get_secrets_manager()
        
        # Get the database credentials
        db_credentials = secrets_manager.get_db_credentials()
        
        # Check if the credentials are valid
        required_keys = ['postgres_user', 'postgres_password', 'postgres_db', 'postgres_schema']
        
        for key in required_keys:
            if key not in db_credentials:
                print(f"Error: '{key}' not found in db_credentials")
                return False
        
        # Print the credentials (masked for security)
        masked_creds = {k: v[:2] + '*****' if k.endswith('password') else v for k, v in db_credentials.items()}
        print(f"Database credentials: {json.dumps(masked_creds, indent=2)}")
        
        print("✅ Secrets manager can read the secrets")
        return True
    
    except Exception as e:
        print(f"Error testing secrets manager: {e}")
        return False

def test_ci_secrets_extraction():
    """Test that we can extract secrets for CI/CD"""
    print("\n=== Testing CI Secrets Extraction ===")
    
    # Get the path to the secrets file and password file
    script_dir = Path(__file__).parent
    secrets_file = script_dir / "ansible" / "secrets.yml"
    password_file = script_dir / "ansible" / "vault_password.txt"
    
    # Create a temporary directory for the secrets
    temp_dir = script_dir / "temp_secrets"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # View the secrets file
        view_result = run_command(f"ansible-vault view {secrets_file} --vault-password-file {password_file}")
        if view_result is None:
            print("Error: Failed to decrypt the secrets file")
            return False
        
        # Parse the YAML
        try:
            secrets = yaml.safe_load(view_result)
            db_creds = secrets['db_credentials']
            
            # Write the secrets to files
            for key, value in db_creds.items():
                with open(temp_dir / key, 'w') as f:
                    f.write(str(value))
            
            # Check if the files were created
            required_keys = ['postgres_user', 'postgres_password', 'postgres_db', 'postgres_schema']
            for key in required_keys:
                file_path = temp_dir / key
                if not file_path.exists():
                    print(f"Error: Secret file '{key}' was not created")
                    return False
                
                # Check if the content is correct
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if content != str(db_creds[key]):
                        print(f"Error: Secret file '{key}' has incorrect content")
                        return False
            
            print("✅ CI secrets extraction works correctly")
            return True
            
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            return False
    
    finally:
        # Clean up the temporary directory
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def main():
    """Run all tests"""
    print("=== Ansible Vault Setup Tests ===")
    
    # Run the tests
    encryption_test = test_vault_encryption()
    decryption_test = test_vault_decryption()
    manager_test = test_secrets_manager()
    ci_extraction_test = test_ci_secrets_extraction()
    
    # Print the results
    print("\n=== Test Results ===")
    print(f"Vault Encryption Test: {'✅ PASSED' if encryption_test else '❌ FAILED'}")
    print(f"Vault Decryption Test: {'✅ PASSED' if decryption_test else '❌ FAILED'}")
    print(f"Secrets Manager Test: {'✅ PASSED' if manager_test else '❌ FAILED'}")
    print(f"CI Secrets Extraction Test: {'✅ PASSED' if ci_extraction_test else '❌ FAILED'}")
    
    # Overall result
    if encryption_test and decryption_test and manager_test and ci_extraction_test:
        print("\n✅ All tests passed! Your Ansible Vault setup is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
