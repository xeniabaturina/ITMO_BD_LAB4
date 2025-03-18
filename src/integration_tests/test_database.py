import pytest
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from src.database import Base, PredictionResult

# Test database configuration
TEST_DB_USER = os.getenv("POSTGRES_USER")
TEST_DB_PASS = os.getenv("POSTGRES_PASSWORD")
TEST_DB_NAME = os.getenv("POSTGRES_DB")
TEST_DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
TEST_DB_PORT = os.getenv("POSTGRES_PORT", "5432")
TEST_DB_SCHEMA = os.getenv("POSTGRES_SCHEMA", "test")

# Create test database URL
TEST_DATABASE_URL = f"postgresql://{TEST_DB_USER}:{TEST_DB_PASS}@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"


@pytest.fixture(scope="function")
def test_db():
    """Create a test database and tables"""
    # Set the schema in the environment variable
    os.environ["POSTGRES_SCHEMA"] = TEST_DB_SCHEMA

    # Determine if we should use SQLite or PostgreSQL
    use_sqlite = os.environ.get("TESTING") == "1"

    if use_sqlite:
        # Use SQLite in-memory database for tests
        engine = create_engine("sqlite:///:memory:")
    else:
        # Use PostgreSQL with schema for tests
        engine = create_engine(TEST_DATABASE_URL)

        # Set the schema for PostgreSQL
        @event.listens_for(engine, "connect")
        def set_search_path(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute(f"SET search_path TO {TEST_DB_SCHEMA}")
            cursor.close()

        # Create schema if it doesn't exist
        with engine.connect() as conn:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {TEST_DB_SCHEMA}")
            conn.execute("COMMIT")

    # Create all tables
    Base.metadata.create_all(engine)

    # Create session
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()

    yield db

    # Cleanup
    db.close()
    Base.metadata.drop_all(engine)


def test_create_prediction(test_db):
    """Test creating a new prediction record"""
    # Create test prediction
    prediction = PredictionResult(
        culmen_length_mm=39.1,
        culmen_depth_mm=18.7,
        flipper_length_mm=181.0,
        body_mass_g=3750.0,
        predicted_species="Adelie",
        confidence=0.95,
    )

    # Add and commit
    test_db.add(prediction)
    test_db.commit()
    test_db.refresh(prediction)

    # Verify
    assert prediction.id is not None
    assert prediction.predicted_species == "Adelie"
    assert prediction.confidence == 0.95


def test_retrieve_predictions(test_db):
    """Test retrieving predictions"""
    # Create multiple test predictions
    predictions = [
        PredictionResult(
            culmen_length_mm=39.1,
            culmen_depth_mm=18.7,
            flipper_length_mm=181.0,
            body_mass_g=3750.0,
            predicted_species="Adelie",
            confidence=0.95,
        ),
        PredictionResult(
            culmen_length_mm=42.3,
            culmen_depth_mm=19.2,
            flipper_length_mm=190.0,
            body_mass_g=4100.0,
            predicted_species="Gentoo",
            confidence=0.88,
        ),
    ]

    # Add all predictions
    for pred in predictions:
        test_db.add(pred)
    test_db.commit()

    # Retrieve all predictions
    db_predictions = test_db.query(PredictionResult).all()

    # Verify
    assert len(db_predictions) == 2
    assert db_predictions[0].predicted_species in ["Adelie", "Gentoo"]
    assert db_predictions[1].predicted_species in ["Adelie", "Gentoo"]
