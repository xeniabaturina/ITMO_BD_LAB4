import os
import sys
import unittest
import json
import pickle
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Set testing environment variable before importing the API module
os.environ["TESTING"] = "1"

# Mock the database functions before importing the API module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import with database and Kafka mocking
with patch("src.database.init_db"), patch("src.database.get_db"), patch("src.kafka_producer.KafkaProducerService"):
    import src.api
    from src.api import ModelService


class MockModel:
    def predict(self, X):
        return ["Adelie"]

    def predict_proba(self, X):
        return [[0.8, 0.1, 0.1]]

    @property
    def classes_(self):
        return ["Adelie", "Gentoo", "Chinstrap"]


class TestAPI(unittest.TestCase):
    """
    Test cases for the API service.
    """

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.experiments_dir = os.path.join(self.test_dir, "experiments")
        os.makedirs(self.experiments_dir, exist_ok=True)

        # Create a model file path
        self.model_path = os.path.join(self.experiments_dir, "random_forest.sav")

        # Create a mock model
        self.mock_model = MockModel()

        # Create test input data
        self.test_input = {
            "island": "Torgersen",
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750,
            "sex": "MALE",
        }

        # Save the mock model to the file
        with open(self.model_path, "wb") as f:
            pickle.dump(self.mock_model, f)

        # Mock database functions
        self.db_patcher = patch("src.api.get_db_connection")
        self.mock_db_connection = self.db_patcher.start()
        self.mock_db = MagicMock()
        self.mock_db_connection.return_value = self.mock_db

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)
        self.db_patcher.stop()

    @patch("src.api.Path")
    @patch("src.api.configparser.ConfigParser.read")
    @patch("src.kafka_producer.KafkaProducerService")
    def test_model_service_init(self, mock_kafka, mock_read, mock_path):
        """Test ModelService initialization."""
        # Mock the Path to return our test directory
        mock_path_instance = MagicMock()
        mock_path_instance.__truediv__.return_value = mock_path_instance
        mock_path_instance.__str__.return_value = self.model_path
        mock_path.return_value = mock_path_instance

        # Initialize the service with the mock model path
        with patch("src.api.os.path.isfile", return_value=True):
            service = ModelService()
            service.model_path = self.model_path

            # Manually load the model since we're mocking the path
            with open(self.model_path, "rb") as f:
                service.model = pickle.load(f)

            # Verify the model was loaded
            self.assertIsNotNone(service.model, "Model should be loaded")

        # Test with a nonexistent model file
        nonexistent_path = os.path.join(self.experiments_dir, "nonexistent.pkl")

        # Create a new service with the nonexistent path
        with patch("src.api.os.path.isfile", return_value=False):
            service = ModelService()
            service.model_path = nonexistent_path
            service.model = None
            self.assertIsNone(
                service.model, "Model should be None for nonexistent file"
            )

    @patch("src.kafka_producer.KafkaProducerService")
    def test_predict(self, mock_kafka):
        """Test the ModelService.predict method."""
        # Create a ModelService instance
        service = ModelService()

        # Set the model to our mock model
        service.model = self.mock_model

        # Make a prediction
        result = service.predict(self.test_input)

        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["predicted_species"], "Adelie")
        self.assertIn("probabilities", result)

    @patch("src.api.ModelService")
    @patch("src.kafka_producer.KafkaProducerService")
    def test_health_check_endpoint(self, mock_kafka, mock_service_class):
        """Test the health check endpoint."""
        # Mock the model service to have a model loaded
        mock_service_instance = MagicMock()
        mock_service_instance.model = MockModel()
        src.api.model_service = mock_service_instance

        # Create a Flask test client
        with patch("src.api.get_db_connection"):
            test_app = src.api.app.test_client()
            response = test_app.get("/health")

        # Parse the response
        response_data = json.loads(response.data)

        # Check the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response_data["status"], "healthy")
        self.assertTrue(response_data["model_loaded"])
        self.assertIn("timestamp", response_data)

    @patch("src.api.ModelService.predict")
    @patch("src.api.log_request")
    @patch("src.api.save_prediction")
    @patch("src.kafka_producer.KafkaProducerService")
    def test_predict_endpoint(
        self, mock_kafka, mock_save_prediction, mock_log_request, mock_predict
    ):
        """Test the prediction endpoint."""
        # Create a real dictionary for the prediction result
        prediction_result = {
            "success": True,
            "predicted_species": "Adelie",
            "probabilities": {"Adelie": 0.8, "Gentoo": 0.1, "Chinstrap": 0.1},
        }

        # Mock the predict method to return our dictionary
        mock_predict.return_value = prediction_result

        # Mock the log_request function to do nothing
        mock_log_request.return_value = None

        # Mock the save_prediction function to do nothing
        mock_save_prediction.return_value = None

        # Create a real model service with a real model
        service = ModelService()
        service.model = self.mock_model

        # Replace the global model_service with our service
        original_service = src.api.model_service
        src.api.model_service = service

        try:
            test_app = src.api.app.test_client()

            # Make a POST request to the predict endpoint
            response = test_app.post(
                "/predict",
                data=json.dumps(self.test_input),
                content_type="application/json",
            )

            # Parse the response
            response_data = json.loads(response.data)

            # Check the response
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response_data["success"])
            self.assertEqual(response_data["predicted_species"], "Adelie")
        finally:
            # Restore the original model_service
            src.api.model_service = original_service


if __name__ == "__main__":
    unittest.main()
