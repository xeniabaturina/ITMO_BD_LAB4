import sys


try:
    from src.secrets_manager import get_secrets_manager
    
    # Get credentials from vault
    sm = get_secrets_manager()
    creds = sm.get_db_credentials()
    
    # Write to environment file
    with open('/shared/db_credentials.env', 'w') as f:
        f.write(f"POSTGRES_USER={creds['postgres_user']}\n")
        f.write(f"POSTGRES_PASSWORD={creds['postgres_password']}\n")
        f.write(f"POSTGRES_DB={creds['postgres_db']}\n")
    
    print("Database credentials extracted successfully")
    
except Exception as e:
    print(f"Error extracting credentials: {e}")
    sys.exit(1)
