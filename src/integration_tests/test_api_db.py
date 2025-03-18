import json
import pytest
import os
from src.api import app
from src.database import init_db
import time

# Test database configuration
TEST_DB_USER = os.getenv("POSTGRES_USER")
TEST_DB_PASS = os.getenv("POSTGRES_PASSWORD")
TEST_DB_NAME = os.getenv("POSTGRES_DB")
TEST_DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
TEST_DB_PORT = os.getenv("POSTGRES_PORT", "5432")
TEST_DB_SCHEMA = os.getenv("POSTGRES_SCHEMA", "test")

# Create test database URL
TEST_DATABASE_URL = f"postgresql://{TEST_DB_USER}:{TEST_DB_PASS}@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"


@pytest.fixture
def client():
    """Create a test client"""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(scope="function")
def test_db():
    """Set up test database"""
    # Set test environment variables for the Flask app
    os.environ["POSTGRES_USER"] = TEST_DB_USER
    os.environ["POSTGRES_PASSWORD"] = TEST_DB_PASS
    os.environ["POSTGRES_DB"] = TEST_DB_NAME
    os.environ["POSTGRES_HOST"] = TEST_DB_HOST
    os.environ["POSTGRES_PORT"] = TEST_DB_PORT
    os.environ["POSTGRES_SCHEMA"] = TEST_DB_SCHEMA
    os.environ["TESTING"] = "1"  # Set testing mode

    # When TESTING=1, we use SQLite in-memory database
    # So we don't need to create a PostgreSQL engine or set up schemas

    # Initialize the database (this will use SQLite in-memory)
    init_db()

    yield


def test_predict_endpoint(client, test_db):
    """Test the predict endpoint with database integration"""
    # Test data
    test_data = {
        "island": "Torgersen",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "sex": "MALE",
    }

    # Make prediction request
    response = client.post(
        "/predict", data=json.dumps(test_data), content_type="application/json"
    )

    # Check response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert "predicted_species" in data
    assert "probabilities" in data


def test_predictions_endpoint(client, test_db):
    """Test the predictions endpoint"""
    print("Starting test_predictions_endpoint")

    # First make some predictions
    test_data = [
        {
            "island": "Torgersen",
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "MALE",
        },
        {
            "island": "Dream",
            "bill_length_mm": 42.3,
            "bill_depth_mm": 19.2,
            "flipper_length_mm": 190.0,
            "body_mass_g": 4100.0,
            "sex": "FEMALE",
        },
    ]

    # Make predictions
    print("Making predictions...")
    for i, data in enumerate(test_data):
        print(f"Making prediction {i + 1}/{len(test_data)}")
        response = client.post(
            "/predict", data=json.dumps(data), content_type="application/json"
        )
        assert response.status_code == 200
        print(f"Prediction {i + 1} response: {response.data}")

    # Test getting predictions
    print("Getting predictions...")
    start_time = time.time()

    try:
        response = client.get("/predictions")
        print(f"Got predictions response in {time.time() - start_time:.2f} seconds")
        print(f"Response status: {response.status_code}")
        print(f"Response data: {response.data}")

        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "predictions" in data
        print(f"Number of predictions: {len(data['predictions'])}")
        assert len(data["predictions"]) >= 1  # We made at least 1 prediction

        # Test pagination
        print("Testing pagination...")
        response = client.get("/predictions?limit=1&offset=0")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["predictions"]) == 1
        print("Test completed successfully")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback

        traceback.print_exc()
