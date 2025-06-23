import os
import yaml
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class SecretsManager:
    def __init__(self, vault_file_path=None, vault_password=None):
        """
        Initialize the SecretsManager with path to the vault file and the vault password.

        Args:
            vault_file_path: Path to the encrypted Ansible Vault file
            vault_password: The Ansible Vault password (if provided directly)
        """
        # Default path if not provided
        self.vault_file_path = vault_file_path or os.environ.get(
            "VAULT_FILE_PATH", "/app/ansible/secrets.yml"
        )

        # Validate vault file path
        self._validate_vault_file_path()

        # Get vault password from environment variable or parameter
        self.vault_password = vault_password or os.environ.get("VAULT_PASSWORD")

        # Ensure we have a vault password
        if not self.vault_password:
            logger.error(
                "Vault password not provided and VAULT_PASSWORD environment variable not set"
            )
            raise ValueError(
                "Vault password not provided and VAULT_PASSWORD environment variable not set"
            )

    def _validate_vault_file_path(self):
        """
        Validate the vault file path for security and existence.
        """
        vault_path = Path(self.vault_file_path)
        
        # Check if path exists
        if not vault_path.exists():
            logger.error(f"Vault file not found at {self.vault_file_path}")
            raise FileNotFoundError(f"Vault file not found at {self.vault_file_path}")
        
        # Security validation: ensure path is within expected directories
        resolved_path = vault_path.resolve()
        allowed_dirs = [
            Path("/app/ansible").resolve(),
            Path("./ansible").resolve(),
            Path("ansible").resolve(),
        ]
        
        path_is_safe = any(
            str(resolved_path).startswith(str(allowed_dir)) 
            for allowed_dir in allowed_dirs
        )
        
        if not path_is_safe:
            logger.error(f"Vault file path {self.vault_file_path} is not in allowed directories")
            raise ValueError(f"Vault file path {self.vault_file_path} is not in allowed directories")
        
        # Check file permissions (should be readable)
        if not os.access(vault_path, os.R_OK):
            logger.error(f"Vault file {self.vault_file_path} is not readable")
            raise PermissionError(f"Vault file {self.vault_file_path} is not readable")
        
        logger.info(f"Vault file path validated: {self.vault_file_path}")

    def get_secrets(self):
        """
        Retrieve secrets from the Ansible Vault using Python libraries.

        Returns:
            dict: Dictionary containing the decrypted secrets
        """
        try:
            # Try to import ansible-vault library first
            try:
                from ansible.parsing.vault import VaultLib
                from ansible.parsing.vault import VaultSecret
                return self._decrypt_with_ansible_lib()
            except ImportError:
                # Fallback to reading if file is not encrypted or using alternative approach
                logger.warning("Ansible library not available, trying alternative decryption")
                return self._decrypt_alternative()
                
        except Exception as e:
            logger.error(f"Unexpected error retrieving secrets: {e}")
            raise

    def _decrypt_with_ansible_lib(self):
        """Decrypt using ansible library."""
        from ansible.parsing.vault import VaultLib
        from ansible.parsing.vault import VaultSecret
        
        # Create vault secret
        vault_secret = VaultSecret(self.vault_password.encode())
        vault = VaultLib([(b'default', vault_secret)])
        
        # Read and decrypt the vault file
        with open(self.vault_file_path, 'rb') as vault_file:
            encrypted_data = vault_file.read()
        
        # Decrypt the data
        decrypted_data = vault.decrypt(encrypted_data)
        
        # Parse the YAML
        secrets = yaml.safe_load(decrypted_data.decode())
        return secrets

    def _decrypt_alternative(self):
        """
        Alternative decryption method if ansible library is not available.
        This assumes the file might not be encrypted or uses a simple format.
        """
        try:
            with open(self.vault_file_path, 'r') as file:
                content = file.read()
                
            # Check if file is encrypted (starts with $ANSIBLE_VAULT)
            if content.startswith('$ANSIBLE_VAULT'):
                # For production, should use proper Ansible Vault library
                # This is a fallback that requires manual decryption
                logger.error("File is encrypted but ansible library not available")
                raise RuntimeError(
                    "Encrypted vault file requires ansible library. "
                    "Please install ansible: pip install ansible"
                )
            else:
                # File is not encrypted, parse directly
                logger.info("Reading unencrypted secrets file")
                secrets = yaml.safe_load(content)
                return secrets
                
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse secrets YAML: {e}")
            raise RuntimeError(f"Failed to parse secrets YAML: {e}")
        except Exception as e:
            logger.error(f"Failed to read secrets file: {e}")
            raise RuntimeError(f"Failed to read secrets file: {e}")

    def get_db_credentials(self):
        """
        Get database credentials from the vault.

        Returns:
            dict: Dictionary containing database credentials
        """
        secrets = self.get_secrets()
        return secrets.get("db_credentials", {})


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
