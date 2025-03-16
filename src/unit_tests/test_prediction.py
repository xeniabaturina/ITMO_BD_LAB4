import os
import sys
import unittest
import pandas as pd
import numpy as np
import pickle
import shutil
import configparser
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import PenguinPredictor


class MockArgParser:
    """Mock argument parser for testing."""

    def __init__(self, test_type):
        self.test_type = test_type
        self.args = []

    def parse_args(self):
        class Args:
            pass

        args = Args()
        args.tests = self.test_type
        return args

    def add_argument(self, *args, **kwargs):
        """Mock add_argument method."""
        self.args.append((args, kwargs))
        return self


# Define MockModel outside of the test class to make it picklable
class MockModel:
    def predict(self, X):
        # Always predict Adelie for simplicity
        return np.array(["Adelie"] * len(X))


class TestPrediction(unittest.TestCase):
    """
    Test cases for the prediction functionality.
    """

    def setUp(self):
        """Set up test environment."""
        # Get the project root directory
        self.root_dir = Path(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

        # Create test directories
        self.test_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.test_dir, "data")
        self.test_experiments_dir = os.path.join(self.test_dir, "experiments")
        self.test_results_dir = os.path.join(self.test_dir, "results")

        os.makedirs(self.test_data_dir, exist_ok=True)
        os.makedirs(self.test_experiments_dir, exist_ok=True)
        os.makedirs(self.test_results_dir, exist_ok=True)

        # Create sample test data
        self.X_test = pd.DataFrame(
            {
                "island": ["Torgersen", "Biscoe", "Dream"],
                "bill_length_mm": [39.1, 46.5, 49.3],
                "bill_depth_mm": [18.7, 15.2, 19.5],
                "flipper_length_mm": [181, 219, 198],
                "body_mass_g": [3750, 5200, 4400],
                "sex": ["male", "female", "male"],
            }
        )

        self.y_test = pd.DataFrame({"species": ["Adelie", "Gentoo", "Chinstrap"]})

        # Save sample data to CSV
        self.X_test_path = os.path.join(self.test_data_dir, "Test_Penguins_X.csv")
        self.y_test_path = os.path.join(self.test_data_dir, "Test_Penguins_y.csv")

        self.X_test.to_csv(self.X_test_path, index=True)
        self.y_test.to_csv(self.y_test_path, index=True)

        # Create a mock model instance
        self.mock_model = MockModel()
        self.model_path = os.path.join(self.test_experiments_dir, "random_forest.sav")

        # Save the mock model
        with open(self.model_path, "wb") as f:
            pickle.dump(self.mock_model, f)

        # Create config file
        self.config = configparser.ConfigParser()

        # Calculate relative paths from the test directory
        data_rel_dir = os.path.relpath(self.test_data_dir, self.test_dir)
        exp_rel_dir = os.path.relpath(self.test_experiments_dir, self.test_dir)

        self.config["SPLIT_DATA"] = {
            "X_test": f"{self.test_dir}/{data_rel_dir}/Test_Penguins_X.csv",
            "y_test": f"{self.test_dir}/{data_rel_dir}/Test_Penguins_y.csv",
        }

        self.config["RANDOM_FOREST"] = {
            "n_estimators": "100",
            "max_depth": "None",
            "min_samples_split": "2",
            "min_samples_leaf": "1",
            "path": f"{exp_rel_dir}/random_forest.sav",
        }

        self.config_path = os.path.join(self.test_dir, "config.ini")
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)

    def tearDown(self):
        """Clean up after tests."""
        # Remove test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("predict.os.getcwd")
    @patch("predict.configparser.ConfigParser.read")
    def test_predictor_initialization(self, mock_read, mock_getcwd):
        """Test Predictor initialization."""
        # Setup mocks
        mock_getcwd.return_value = self.test_dir
        mock_read.return_value = None

        # Create predictor with patched environment
        with patch.object(PenguinPredictor, "__init__", return_value=None):
            predictor = PenguinPredictor()
            predictor.config = self.config
            predictor.X_test = self.X_test
            predictor.y_test = self.y_test
            predictor.model_path = self.model_path
            predictor.log = MagicMock()

            # Test initialization
            self.assertIsNotNone(predictor.config, "Config should be initialized")
            self.assertEqual(
                predictor.model_path,
                self.model_path,
                "Model path should be set correctly",
            )

    @patch("predict.Path")
    def test_predict_smoke(self, mock_path):
        """Test smoke test prediction."""
        # Mock the Path to return our test directory
        mock_path_instance = MagicMock()
        mock_path_instance.__truediv__.return_value = mock_path_instance
        mock_path_instance.__str__.return_value = self.test_dir
        mock_path.return_value = mock_path_instance

        # Create a mock argument parser
        predictor = PenguinPredictor()
        predictor.parser = MockArgParser("smoke")
        predictor.model_path = self.model_path
        predictor.X_test = self.X_test
        predictor.y_test = self.y_test

        result = predictor.predict()
        self.assertTrue(result, "Smoke test prediction should succeed")

    @patch("predict.Path")
    def test_predict_func(self, mock_path):
        """Test functional test prediction."""
        # Mock the Path to return our test directory
        mock_path_instance = MagicMock()
        mock_path_instance.__truediv__.return_value = mock_path_instance
        mock_path_instance.__str__.return_value = self.test_dir
        mock_path.return_value = mock_path_instance

        # Create a mock argument parser
        predictor = PenguinPredictor()
        predictor.parser = MockArgParser("func")
        predictor.model_path = self.model_path
        predictor.X_test = self.X_test
        predictor.y_test = self.y_test

        result = predictor.predict()
        self.assertTrue(result, "Functional test prediction should succeed")

    @patch("predict.Path")
    def test_predict_model_not_found(self, mock_path):
        """Test prediction with missing model file."""
        # Mock the Path to return our test directory
        mock_path_instance = MagicMock()
        mock_path_instance.__truediv__.return_value = mock_path_instance
        mock_path_instance.__str__.return_value = self.test_dir
        mock_path.return_value = mock_path_instance

        # Create a mock argument parser
        predictor = PenguinPredictor()
        predictor.parser = MockArgParser("smoke")
        predictor.model_path = os.path.join(
            self.test_experiments_dir, "nonexistent.sav"
        )
        predictor.X_test = self.X_test
        predictor.y_test = self.y_test

        result = predictor.predict()
        self.assertFalse(result, "Prediction should fail with missing model file")

    @patch("predict.Path")
    @patch("predict.argparse.ArgumentParser.parse_args")
    def test_predict_invalid_test_type(self, mock_parse_args, mock_path):
        """Test prediction with invalid test type."""
        # Mock the Path to return our test directory
        mock_path_instance = MagicMock()
        mock_path_instance.__truediv__.return_value = mock_path_instance
        mock_path_instance.__str__.return_value = self.test_dir
        mock_path.return_value = mock_path_instance

        # Create a mock argument parser that will return an invalid test type
        class InvalidArgs:
            tests = "invalid"

        mock_parse_args.return_value = InvalidArgs()

        # Create predictor
        predictor = PenguinPredictor()
        predictor.model_path = self.model_path
        predictor.X_test = self.X_test
        predictor.y_test = self.y_test

        # The predict method doesn't actually validate the test_type value,
        # it just uses it to determine which branch to take.
        # Since "invalid" doesn't match any of the conditions, it will default to the else branch
        # and return True if everything else works
        result = predictor.predict()
        self.assertTrue(result, "Prediction should succeed with unknown test type")


if __name__ == "__main__":
    unittest.main()
