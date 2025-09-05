import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from .secrets_manager import get_secrets_manager

# Create base class for declarative models
Base = declarative_base()

# Get database connection details from secrets manager (or environment variables for testing)
if os.environ.get("TESTING") == "1":
    # Use environment variables for testing
    DB_USER = os.getenv("POSTGRES_USER")
    DB_PASS = os.getenv("POSTGRES_PASSWORD")
    DB_NAME = os.getenv("POSTGRES_DB")
    DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    DB_SCHEMA = os.getenv("POSTGRES_SCHEMA", "public")
else:
    # Get credentials from Ansible Vault
    try:
        secrets_manager = get_secrets_manager()
        db_credentials = secrets_manager.get_db_credentials()

        DB_USER = db_credentials.get("postgres_user")
        DB_PASS = db_credentials.get("postgres_password")
        DB_NAME = db_credentials.get("postgres_db")
        DB_SCHEMA = db_credentials.get("postgres_schema", "public")

        # These can still come from environment variables as they're not sensitive
        DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
        DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    except Exception as e:
        # Log the error and fall back to environment variables
        print(f"Error retrieving secrets from vault: {e}")
        print("Falling back to environment variables")

        DB_USER = os.getenv("POSTGRES_USER")
        DB_PASS = os.getenv("POSTGRES_PASSWORD")
        DB_NAME = os.getenv("POSTGRES_DB")
        DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
        DB_PORT = os.getenv("POSTGRES_PORT", "5432")
        DB_SCHEMA = os.getenv("POSTGRES_SCHEMA", "public")

# Create database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine and session factory
engine = None
SessionLocal = None


class PredictionResult(Base):
    __tablename__ = "prediction_results"
    # Use the schema only for PostgreSQL
    __table_args__ = (
        {"schema": DB_SCHEMA} if not os.environ.get("TESTING") == "1" else {}
    )

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Input features
    culmen_length_mm = Column(Float)
    culmen_depth_mm = Column(Float)
    flipper_length_mm = Column(Float)
    body_mass_g = Column(Float)

    # Prediction
    predicted_species = Column(String)
    confidence = Column(Float)


def init_db():
    """Initialize database connection and create tables"""
    global engine, SessionLocal

    # Check if we're in a test environment
    if os.environ.get("TESTING") == "1":
        # Use SQLite in-memory database for tests
        engine = create_engine("sqlite:///:memory:")
    else:
        # Use PostgreSQL for production
        engine = create_engine(DATABASE_URL)

        # Set the schema for PostgreSQL
        @event.listens_for(engine, "connect")
        def set_search_path(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute(f"SET search_path TO {DB_SCHEMA}")
            cursor.close()

    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    # Initialize database if not already initialized
    if engine is None or SessionLocal is None:
        init_db()

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(
    db,
    culmen_length_mm: float,
    culmen_depth_mm: float,
    flipper_length_mm: float,
    body_mass_g: float,
    predicted_species: str,
    confidence: float,
):
    prediction = PredictionResult(
        culmen_length_mm=culmen_length_mm,
        culmen_depth_mm=culmen_depth_mm,
        flipper_length_mm=flipper_length_mm,
        body_mass_g=body_mass_g,
        predicted_species=predicted_species,
        confidence=confidence,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction.id
