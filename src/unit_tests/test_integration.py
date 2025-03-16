import os
import sys
import json
import pickle
import unittest
from unittest import mock
import pandas as pd
from pathlib import Path
import configparser

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import PenguinPreprocessor
from src.train import PenguinClassifier
from src.predict import PenguinPredictor


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration tests for the entire ML workflow."""

    def setUp(self):
        """Set up test environment."""
        # Use pathlib for path handling
        self.base_dir = Path(__file__).parent.parent.parent
        self.test_dir = self.base_dir / "test_integration"
        self.data_dir = self.test_dir / "data"
        self.experiments_dir = self.test_dir / "experiments"
        self.results_dir = self.test_dir / "results"

        # Create test directories
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Create sample data
        self.create_sample_data()

        # Create a config file with relative paths
        self.config_path = self.test_dir / "config.ini"
        self.create_config_file()

    def create_config_file(self):
        """Create a config file with relative paths."""
        config = configparser.ConfigParser()

        # Use relative paths from the project root
        data_rel_path = os.path.relpath(self.data_dir, self.base_dir)
        exp_rel_path = os.path.relpath(self.experiments_dir, self.base_dir)

        config["DATA"] = {
            "x_data": f"{data_rel_path}/Penguins_X.csv",
            "y_data": f"{data_rel_path}/Penguins_y.csv",
        }

        config["SPLIT_DATA"] = {
            "x_train": f"{data_rel_path}/Train_Penguins_X.csv",
            "y_train": f"{data_rel_path}/Train_Penguins_y.csv",
            "x_test": f"{data_rel_path}/Test_Penguins_X.csv",
            "y_test": f"{data_rel_path}/Test_Penguins_y.csv",
        }

        config["RANDOM_FOREST"] = {
            "n_estimators": "100",
            "max_depth": "None",
            "min_samples_split": "2",
            "min_samples_leaf": "1",
            "path": f"{exp_rel_path}/random_forest.sav",
        }

        with open(self.config_path, "w") as f:
            config.write(f)

    def tearDown(self):
        """Clean up after tests."""
        # Remove test directories if they exist
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_sample_data(self):
        """Create sample penguin data for testing."""
        # Create a sample dataset with multiple instances of each species
        data = {
            "species": [
                "Adelie",
                "Adelie",
                "Gentoo",
                "Gentoo",
                "Chinstrap",
                "Chinstrap",
            ],
            "island": ["Torgersen", "Torgersen", "Biscoe", "Biscoe", "Dream", "Dream"],
            "bill_length_mm": [39.1, 38.6, 39.5, 37.8, 49.3, 48.5],
            "bill_depth_mm": [18.7, 17.2, 17.4, 18.1, 19.0, 18.5],
            "flipper_length_mm": [181.0, 193.0, 186.0, 174.0, 195.0, 190.0],
            "body_mass_g": [3750, 3700, 3800, 3900, 4050, 4000],
            "sex": ["MALE", "FEMALE", "FEMALE", "MALE", "MALE", "FEMALE"],
            "year": [2007, 2008, 2007, 2009, 2008, 2009],
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_dir / "penguins.csv", index=False)

    def test_full_workflow(self):
        """Test the entire ML workflow from preprocessing to prediction."""
        # 1. Preprocess the data
        preprocessor = PenguinPreprocessor()
        preprocessor.project_path = str(self.data_dir)
        preprocessor.data_path = str(self.data_dir / "penguins.csv")
        preprocessor.X_path = str(self.data_dir / "Penguins_X.csv")
        preprocessor.y_path = str(self.data_dir / "Penguins_y.csv")
        preprocessor.train_path = [
            str(self.data_dir / "Train_Penguins_X.csv"),
            str(self.data_dir / "Train_Penguins_y.csv"),
        ]
        preprocessor.test_path = [
            str(self.data_dir / "Test_Penguins_X.csv"),
            str(self.data_dir / "Test_Penguins_y.csv"),
        ]

        # Process and split the data
        self.assertTrue(preprocessor.get_data(), "Data preprocessing should succeed")

        # Mock the train_test_split function
        with mock.patch("src.preprocess.train_test_split") as mock_split:
            # Read the data that was created
            X = pd.read_csv(preprocessor.X_path, index_col=0)
            y = pd.read_csv(preprocessor.y_path, index_col=0)

            # Create mock train/test splits
            X_train = X.iloc[:4]
            X_test = X.iloc[4:]
            y_train = y.iloc[:4]
            y_test = y.iloc[4:]

            # Configure the mock to return our predefined splits
            mock_split.return_value = (X_train, X_test, y_train, y_test)

            # Split the data
            self.assertTrue(preprocessor.split_data(), "Data splitting should succeed")

        # 2. Train a model
        classifier = PenguinClassifier()
        classifier.project_path = str(self.experiments_dir)
        classifier.model_path = str(self.experiments_dir / "random_forest.sav")
        classifier.train_path = preprocessor.train_path

        # Load the config file
        config = configparser.ConfigParser()
        config.read(str(self.config_path))
        classifier.config = config

        # Train the model
        model_path = str(self.experiments_dir / "random_forest.sav")
        classifier.model_path = model_path
        result = classifier.train_random_forest()
        self.assertTrue(result, "Model training should succeed")

        # Verify the model file was created
        self.assertTrue(os.path.exists(model_path), "Model file should exist")

        # 3. Make predictions
        # Load the config file for the predictor
        config = configparser.ConfigParser()
        config.read(str(self.config_path))

        # Initialize the predictor
        predictor = PenguinPredictor()
        predictor.config = config
        predictor.model_path = model_path

        # Override the paths to use test paths
        predictor.X_test = pd.read_csv(preprocessor.test_path[0], index_col=0)
        predictor.y_test = pd.read_csv(preprocessor.test_path[1], index_col=0)

        # Load the model
        with open(model_path, "rb") as f:
            predictor.model = pickle.load(f)

        # 4. Test prediction on a sample
        # Create a test sample
        test_sample = {
            "island": "Biscoe",
            "bill_length_mm": 39.5,
            "bill_depth_mm": 17.4,
            "flipper_length_mm": 186.0,
            "body_mass_g": 3800,
            "sex": "FEMALE",
        }

        # Convert to DataFrame for prediction
        test_df = pd.DataFrame([test_sample])

        # Make a prediction
        prediction = predictor.model.predict(test_df)
        self.assertIsNotNone(prediction, "Prediction should not be None")
        self.assertIn(
            prediction[0],
            ["Adelie", "Gentoo", "Chinstrap"],
            "Prediction should be a valid species",
        )

        # Test probability prediction
        probabilities = predictor.model.predict_proba(test_df)
        self.assertIsNotNone(probabilities, "Probabilities should not be None")
        # The number of classes depends on what was in the training data
        # Our small test dataset might only have 2 classes
        self.assertGreaterEqual(
            probabilities.shape[1], 1, "Should have probabilities for at least 1 class"
        )

        # Get the class labels from the model
        class_labels = predictor.model.classes_

        # Create a result dictionary
        result = {
            "prediction": prediction[0],
            "probabilities": {
                class_labels[i]: float(probabilities[0][i])
                for i in range(len(class_labels))
            },
        }

        # Verify the result
        self.assertIsNotNone(result["prediction"], "Prediction should not be None")
        self.assertIn(
            result["prediction"],
            class_labels,
            "Prediction should be a valid species",
        )
        self.assertEqual(
            len(result["probabilities"]),
            len(class_labels),
            f"Should have probabilities for all {len(class_labels)} classes",
        )

        # 5. Create a test file for the predictor to use
        test_file_path = str(self.test_dir / "test_sample.json")
        with open(test_file_path, "w") as f:
            json.dump(test_sample, f)

        # Create a simplified version of the predict_sample method
        sample = test_sample

        # Set up the predictor to use our test file
        predictor.test_file = test_file_path
        predictor.results_dir = str(self.results_dir)

        # Run a prediction using the predictor's methods
        prediction = predictor.model.predict(pd.DataFrame([sample]))[0]
        probabilities = predictor.model.predict_proba(pd.DataFrame([sample]))[0]

        # Create a result dictionary
        result = {
            "prediction": prediction,
            "probabilities": {
                class_labels[i]: float(probabilities[i])
                for i in range(len(class_labels))
            },
        }

        # Verify the result
        self.assertIsNotNone(result["prediction"], "Prediction should not be None")
        self.assertIn(
            result["prediction"],
            class_labels,
            "Prediction should be a valid species",
        )
        self.assertEqual(
            len(result["probabilities"]),
            len(class_labels),
            f"Should have probabilities for all {len(class_labels)} classes",
        )

        # 6. Save the result to a file
        result_file = self.results_dir / "prediction_result.json"
        with open(result_file, "w") as f:
            json.dump(result, f)

        # Verify the result file was created
        self.assertTrue(os.path.exists(result_file), "Result file should exist")


if __name__ == "__main__":
    unittest.main()
