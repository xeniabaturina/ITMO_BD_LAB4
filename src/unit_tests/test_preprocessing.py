import os
import sys
import unittest
import pandas as pd
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import PenguinPreprocessor


class TestPreprocessing(unittest.TestCase):
    """
    Test cases for the preprocessing functionality.
    """

    def setUp(self):
        """Set up test environment."""
        # Use pathlib for more reliable path handling
        self.base_dir = Path(__file__).parent.parent.parent
        self.test_dir = self.base_dir / "test_preprocessing"

        # Create test data directory
        os.makedirs(self.test_dir, exist_ok=True)

        # Create sample data
        self.create_sample_data()

        # Initialize PenguinPreprocessor with test paths
        self.data_maker = PenguinPreprocessor()
        self.data_maker.project_path = str(self.test_dir)
        self.data_maker.data_path = str(self.test_dir / "penguins.csv")
        self.data_maker.X_path = str(self.test_dir / "Penguins_X.csv")
        self.data_maker.y_path = str(self.test_dir / "Penguins_y.csv")
        self.data_maker.train_path = [
            str(self.test_dir / "Train_Penguins_X.csv"),
            str(self.test_dir / "Train_Penguins_y.csv"),
        ]
        self.data_maker.test_path = [
            str(self.test_dir / "Test_Penguins_X.csv"),
            str(self.test_dir / "Test_Penguins_y.csv"),
        ]

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_sample_data(self):
        """Create sample penguin data for testing."""
        # Create a sample dataset with multiple instances of each species for stratification
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
        df.to_csv(self.test_dir / "penguins.csv", index=False)

    def test_get_data(self):
        """Test data loading and splitting into features and target."""
        result = self.data_maker.get_data()
        self.assertTrue(result, "Data loading should succeed")
        self.assertTrue(
            os.path.exists(self.data_maker.X_path), "X data file should be created"
        )
        self.assertTrue(
            os.path.exists(self.data_maker.y_path), "y data file should be created"
        )

        # Check that X data doesn't contain species or year
        X_data = pd.read_csv(self.data_maker.X_path, index_col=0)
        self.assertNotIn("species", X_data.columns, "X data should not contain species")
        self.assertNotIn("year", X_data.columns, "X data should not contain year")

        # Check that y data contains only species
        y_data = pd.read_csv(self.data_maker.y_path, index_col=0)
        self.assertIn("species", y_data.columns, "y data should contain species")
        self.assertEqual(
            y_data.shape[1], 1, "y data should contain only the species column"
        )

    @patch("src.preprocess.train_test_split")
    def test_split_data(self, mock_train_test_split):
        """Test splitting data into training and testing sets."""
        # First load the data
        self.data_maker.get_data()

        # Read the data that was created
        X = pd.read_csv(self.data_maker.X_path, index_col=0)
        y = pd.read_csv(self.data_maker.y_path, index_col=0)

        # Create mock train/test splits
        X_train = X.iloc[:4]
        X_test = X.iloc[4:]
        y_train = y.iloc[:4]
        y_test = y.iloc[4:]

        # Configure the mock to return our predefined splits
        mock_train_test_split.return_value = (X_train, X_test, y_train, y_test)

        # Patch the stratify parameter check in train_test_split
        with patch(
            "src.preprocess.train_test_split",
            return_value=(X_train, X_test, y_train, y_test),
        ):
            # Then split it
            result = self.data_maker.split_data()
            self.assertTrue(result, "Data splitting should succeed")

    def test_error_handling(self):
        """Test error handling in data processing."""
        # Test with non-existent file
        self.data_maker.data_path = str(self.test_dir / "nonexistent.csv")
        result = self.data_maker.get_data()
        self.assertFalse(result, "Should return False for non-existent file")

        # Test with invalid data
        invalid_file = self.test_dir / "invalid.csv"
        with open(invalid_file, "w") as f:
            f.write("This is not a valid CSV file")
        self.data_maker.data_path = str(invalid_file)
        result = self.data_maker.get_data()
        self.assertFalse(result, "Should return False for invalid data")

    def test_file_writing_errors(self):
        """Test handling of file writing errors."""
        # Create a PenguinPreprocessor with a mocked to_csv method
        with patch.object(PenguinPreprocessor, "__init__", return_value=None):
            data_maker = PenguinPreprocessor()
            data_maker.project_path = str(self.test_dir)
            data_maker.data_path = str(self.test_dir / "penguins.csv")
            data_maker.X_path = str(self.test_dir / "Penguins_X.csv")
            data_maker.y_path = str(self.test_dir / "Penguins_y.csv")
            data_maker.log = MagicMock()

            # Mock the to_csv method to raise an exception
            with patch(
                "pandas.DataFrame.to_csv", side_effect=Exception("Simulated error")
            ):
                # Test get_data with file writing error
                result = data_maker.get_data()
                self.assertFalse(result, "Should return False when file writing fails")

    def test_empty_data(self):
        """Test handling of empty data."""
        # Create empty data file
        empty_file = self.test_dir / "empty.csv"
        pd.DataFrame().to_csv(empty_file, index=False)
        self.data_maker.data_path = str(empty_file)

        # Test get_data with empty data
        result = self.data_maker.get_data()
        self.assertFalse(result, "Should return False for empty data")


if __name__ == "__main__":
    unittest.main()
