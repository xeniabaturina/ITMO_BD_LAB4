import os
from .database import init_db, get_db


def get_db_connection():
    """
    Get database connection
    
    Returns:
        Database session or None if connection fails.
    """
    try:
        init_db()
        return next(get_db())
    except Exception as e:
        print(f"Error getting database connection: {e}")
        return None
