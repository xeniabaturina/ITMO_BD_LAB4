import os
import sys
import unittest
import pandas as pd
import numpy as np
import shutil
import configparser
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import PenguinClassifier


class TestTraining(unittest.TestCase):
    """
    Test cases for the training functionality.
    """

    def setUp(self):
        """Set up test environment."""
        # Get the project root directory
        self.root_dir = Path(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

        # Create test directories
        self.test_dir = str(self.root_dir / "test_training")
        self.test_data_dir = str(Path(self.test_dir) / "data")
        self.test_experiments_dir = str(Path(self.test_dir) / "experiments")
        os.makedirs(self.test_data_dir, exist_ok=True)
        os.makedirs(self.test_experiments_dir, exist_ok=True)

        # Create sample train and test data
        self.X_train = pd.DataFrame(
            {
                "island": ["Torgersen", "Biscoe", "Dream"],
                "bill_length_mm": [39.1, 46.5, 49.3],
                "bill_depth_mm": [18.7, 15.2, 19.5],
                "flipper_length_mm": [181, 219, 198],
                "body_mass_g": [3750, 5200, 4400],
                "sex": ["male", "female", "male"],
            }
        )

        self.y_train = pd.DataFrame({"species": ["Adelie", "Gentoo", "Chinstrap"]})

        self.X_test = pd.DataFrame(
            {
                "island": ["Torgersen", "Biscoe"],
                "bill_length_mm": [38.6, 45.2],
                "bill_depth_mm": [17.2, 14.8],
                "flipper_length_mm": [185, 215],
                "body_mass_g": [3800, 5100],
                "sex": ["female", "male"],
            }
        )

        self.y_test = pd.DataFrame({"species": ["Adelie", "Gentoo"]})

        # Save sample data to CSV
        self.X_train_path = os.path.join(self.test_data_dir, "Train_Penguins_X.csv")
        self.y_train_path = os.path.join(self.test_data_dir, "Train_Penguins_y.csv")
        self.X_test_path = os.path.join(self.test_data_dir, "Test_Penguins_X.csv")
        self.y_test_path = os.path.join(self.test_data_dir, "Test_Penguins_y.csv")

        self.X_train.to_csv(self.X_train_path, index=True)
        self.y_train.to_csv(self.y_train_path, index=True)
        self.X_test.to_csv(self.X_test_path, index=True)
        self.y_test.to_csv(self.y_test_path, index=True)

        # Create config file with relative paths
        self.config = configparser.ConfigParser()

        # Calculate relative paths from the project root
        data_rel_dir = os.path.relpath(self.test_data_dir, self.root_dir)
        exp_rel_dir = os.path.relpath(self.test_experiments_dir, self.root_dir)

        self.config["SPLIT_DATA"] = {
            "X_train": f"{data_rel_dir}/Train_Penguins_X.csv",
            "y_train": f"{data_rel_dir}/Train_Penguins_y.csv",
            "X_test": f"{data_rel_dir}/Test_Penguins_X.csv",
            "y_test": f"{data_rel_dir}/Test_Penguins_y.csv",
        }

        self.config_path = str(self.root_dir / "test_config.ini")
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)

        # Create a test model path
        self.model_path = os.path.join(self.test_experiments_dir, "random_forest.sav")

        # Add RANDOM_FOREST section with relative path
        self.config["RANDOM_FOREST"] = {
            "n_estimators": "100",
            "max_depth": "None",
            "min_samples_split": "2",
            "min_samples_leaf": "1",
            "path": f"{exp_rel_dir}/random_forest.sav",
        }

        # Write updated config
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)

    @patch("train.configparser.ConfigParser.read")
    def create_test_classifier(self, mock_read):
        """Create a test classifier with mocked config and paths."""
        # Mock the read method to do nothing
        mock_read.return_value = None

        # Create a classifier with a mocked __init__ method
        with patch.object(PenguinClassifier, "__init__", return_value=None):
            classifier = PenguinClassifier()

            # Override configurations
            classifier.config = self.config

            # Override data attributes
            classifier.X_train = self.X_train
            classifier.y_train = self.y_train
            classifier.X_test = self.X_test
            classifier.y_test = self.y_test

            # Override model path
            classifier.model_path = self.model_path

            # Add a logger mock
            classifier.log = MagicMock()

            return classifier

    def tearDown(self):
        """Clean up after tests."""
        # Remove test directories
        shutil.rmtree(self.test_data_dir, ignore_errors=True)
        shutil.rmtree(self.test_experiments_dir, ignore_errors=True)

        # Remove test config file
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

    def test_train_random_forest(self):
        """Test train_random_forest functionality."""
        # Create a classifier instance for this test
        with patch("train.configparser.ConfigParser.read"):
            classifier = self.create_test_classifier()

            # Mock the save_model method to return True
            classifier.save_model = MagicMock(return_value=True)

            # Train the model
            result = classifier.train_random_forest()

            # Check result
            self.assertTrue(result, "train_random_forest should return True on success")

            # Verify save_model was called
            classifier.save_model.assert_called_once()

    def test_train_random_forest_with_config(self):
        """Test train_random_forest with config parameters."""
        # Create a classifier instance for this test
        with patch("train.configparser.ConfigParser.read"):
            classifier = self.create_test_classifier()

            # Add RandomForest config section
            classifier.config["RANDOM_FOREST"] = {
                "n_estimators": "100",
                "max_depth": "10",
                "min_samples_split": "2",
                "min_samples_leaf": "1",
            }

            # Mock the save_model method to return True
            classifier.save_model = MagicMock(return_value=True)

            # Train the model
            result = classifier.train_random_forest()

            # Check result
            self.assertTrue(result, "train_random_forest should return True on success")

            # Verify save_model was called
            classifier.save_model.assert_called_once()

    def test_train_random_forest_with_none_max_depth(self):
        """Test train_random_forest with None max_depth."""
        # Create a classifier instance for this test
        with patch("train.configparser.ConfigParser.read"):
            classifier = self.create_test_classifier()

            # Add RandomForest config section with None max_depth
            classifier.config["RANDOM_FOREST"] = {
                "n_estimators": "100",
                "max_depth": "None",
                "min_samples_split": "2",
                "min_samples_leaf": "1",
            }

            # Mock the save_model method to return True
            classifier.save_model = MagicMock(return_value=True)

            # Train the model
            result = classifier.train_random_forest()

            # Check result
            self.assertTrue(result, "train_random_forest should return True on success")

            # Verify save_model was called
            classifier.save_model.assert_called_once()

    def test_save_model(self):
        """Test save_model functionality."""
        # Create a classifier instance for this test
        with patch("train.configparser.ConfigParser.read"):
            classifier = self.create_test_classifier()

            # Add root_dir attribute which is needed by save_model
            classifier.root_dir = Path(self.test_dir)

            # Create a Pipeline model (consistent with training code)
            rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, min_samples_leaf=1)
            model = Pipeline(steps=[("preprocessor", StandardScaler()), ("classifier", rf_classifier)])
            model.fit(
                self.X_train.select_dtypes(include=[np.number]),
                self.y_train.values.ravel(),
            )

            # Mock the open function to avoid actual file writing
            with patch("builtins.open", MagicMock()):
                with patch("pickle.dump", MagicMock()):
                    with patch("os.makedirs", MagicMock()):
                        # Save the model
                        result = classifier.save_model(model)

                        # Check result
                        self.assertTrue(
                            result, "save_model should return True on success"
                        )

    def test_save_model_error(self):
        """Test save_model error handling."""
        # Create a classifier instance for this test
        with patch("train.configparser.ConfigParser.read"):
            classifier = self.create_test_classifier()

            # Add root_dir attribute which is needed by save_model
            classifier.root_dir = Path(self.test_dir)

            # Create a Pipeline model
            rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, min_samples_leaf=1)
            model = Pipeline(steps=[("preprocessor", StandardScaler()), ("classifier", rf_classifier)])
            model.fit(
                self.X_train.select_dtypes(include=[np.number]),
                self.y_train.values.ravel(),
            )

            # Mock the open function to raise an exception
            with patch("builtins.open", side_effect=Exception("Test exception")):
                # Save the model
                result = classifier.save_model(model)

                # Check result
                self.assertFalse(result, "save_model should return False on error")

    @patch("train.Path")
    @patch("train.os.makedirs")
    @patch("train.pd.read_csv")
    def test_classifier_initialization(self, mock_read_csv, mock_makedirs, mock_path):
        """Test PenguinClassifier initialization."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.__truediv__.return_value = mock_path_instance
        mock_path_instance.__str__.return_value = "/test"
        mock_path.return_value = mock_path_instance

        mock_read_csv.side_effect = [
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        ]

        # Create a mock config
        mock_config = configparser.ConfigParser()
        mock_config["SPLIT_DATA"] = {
            "X_train": self.X_train_path,
            "y_train": self.y_train_path,
            "X_test": self.X_test_path,
            "y_test": self.y_test_path,
        }
        # Add RANDOM_FOREST section to the mock config
        mock_config["RANDOM_FOREST"] = {
            "n_estimators": "100",
            "max_depth": "None",
            "min_samples_split": "2",
            "min_samples_leaf": "1",
            "path": "experiments/random_forest.sav",
        }

        # Patch the ConfigParser.read method
        with patch("configparser.ConfigParser.read", return_value=None):
            # Patch the ConfigParser to return our mock config
            with patch("configparser.ConfigParser", return_value=mock_config):
                # Patch the Logger
                with patch("train.Logger", return_value=MagicMock()):
                    # Initialize the classifier
                    classifier = PenguinClassifier()

                    # Verify the classifier was initialized correctly
                    self.assertEqual(classifier.X_train.shape, self.X_train.shape)
                    self.assertEqual(classifier.y_train.shape, self.y_train.shape)
                    self.assertEqual(classifier.X_test.shape, self.X_test.shape)
                    self.assertEqual(classifier.y_test.shape, self.y_test.shape)

                    # Verify the model path was set correctly - we're now using Path
                    self.assertIsNotNone(
                        classifier.model_path, "Model path should not be None"
                    )


if __name__ == "__main__":
    unittest.main()
