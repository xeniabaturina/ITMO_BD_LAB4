import os
import yaml
import subprocess
import logging
import tempfile
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

        # Get vault password from environment variable or parameter
        self.vault_password = vault_password or os.environ.get("VAULT_PASSWORD")

        # Ensure the vault file exists
        if not Path(self.vault_file_path).exists():
            logger.error(f"Vault file not found at {self.vault_file_path}")
            raise FileNotFoundError(f"Vault file not found at {self.vault_file_path}")

        # Ensure we have a vault password
        if not self.vault_password:
            logger.error(
                "Vault password not provided and VAULT_PASSWORD environment variable not set"
            )
            raise ValueError(
                "Vault password not provided and VAULT_PASSWORD environment variable not set"
            )

    def get_secrets(self):
        """
        Retrieve secrets from the Ansible Vault.

        Returns:
            dict: Dictionary containing the decrypted secrets
        """
        try:
            # Create a temporary file for the vault password
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
                temp_file.write(self.vault_password)
                temp_file_path = temp_file.name

            try:
                # Run ansible-vault to decrypt the file
                result = subprocess.run(
                    [
                        "ansible-vault",
                        "view",
                        self.vault_file_path,
                        "--vault-password-file",
                        temp_file_path,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Parse the YAML output
                secrets = yaml.safe_load(result.stdout)
                return secrets
            finally:
                # Always remove the temporary file
                os.unlink(temp_file_path)

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
