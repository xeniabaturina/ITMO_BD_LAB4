import os
import yaml
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SecretsManager:
    def __init__(self, vault_file_path=None, vault_password_file=None):
        """
        Initialize the SecretsManager with paths to the vault file and password file.
        
        Args:
            vault_file_path: Path to the encrypted Ansible Vault file
            vault_password_file: Path to the file containing the Ansible Vault password
        """
        # Default paths if not provided
        self.vault_file_path = vault_file_path or os.environ.get(
            'VAULT_FILE_PATH', '/app/ansible/secrets.yml'
        )
        self.vault_password_file = vault_password_file or os.environ.get(
            'VAULT_PASSWORD_FILE', '/app/ansible/vault_password.txt'
        )
        
        # Ensure the files exist
        if not Path(self.vault_file_path).exists():
            logger.error(f"Vault file not found at {self.vault_file_path}")
            raise FileNotFoundError(f"Vault file not found at {self.vault_file_path}")
        
        if not Path(self.vault_password_file).exists():
            logger.error(f"Vault password file not found at {self.vault_password_file}")
            raise FileNotFoundError(f"Vault password file not found at {self.vault_password_file}")
    
    def get_secrets(self):
        """
        Retrieve secrets from the Ansible Vault.
        
        Returns:
            dict: Dictionary containing the decrypted secrets
        """
        try:
            # Run ansible-vault to decrypt the file
            result = subprocess.run(
                [
                    'ansible-vault', 'view', 
                    self.vault_file_path, 
                    '--vault-password-file', self.vault_password_file
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the YAML output
            secrets = yaml.safe_load(result.stdout)
            return secrets
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to decrypt vault: {e.stderr}")
            raise RuntimeError(f"Failed to decrypt vault: {e.stderr}")
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse secrets YAML: {e}")
            raise RuntimeError(f"Failed to parse secrets YAML: {e}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving secrets: {e}")
            raise

    def get_db_credentials(self):
        """
        Get database credentials from the vault.
        
        Returns:
            dict: Dictionary containing database credentials
        """
        secrets = self.get_secrets()
        return secrets.get('db_credentials', {})


# Singleton instance for application-wide use
_secrets_manager = None

def get_secrets_manager():
    """
    Get or create the SecretsManager singleton instance.
    
    Returns:
        SecretsManager: The secrets manager instance
    """
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager 