import argparse
import configparser
from datetime import datetime
import os
import json
import pandas as pd
import pickle
import numpy as np
import traceback
from pathlib import Path

from src.logger import Logger

SHOW_LOG = True


class PenguinPredictor:
    """
    Class for making predictions with the trained model.
    """

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)

        # Get the project root directory (assuming src is one level below root)
        self.root_dir = Path(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config_path = self.root_dir / "config.ini"
        self.config.read(str(config_path))

        self.parser = argparse.ArgumentParser(description="Penguin Species Predictor")
        self.parser.add_argument(
            "-t",
            "--tests",
            type=str,
            help="Select test type",
            required=True,
            default="smoke",
            const="smoke",
            nargs="?",
            choices=["smoke", "func"],
        )

        self.project_path = str(self.root_dir / "experiments")
        self.model_path = str(Path(self.project_path) / "random_forest.sav")

        # Function to safely load a file, trying both the path from config and a relative fallback
        def safe_load_csv(config_path, relative_fallback):
            try:
                # Try to load using the path from config
                # Convert config_path to a Path object relative to root_dir
                full_path = self.root_dir / config_path
                self.log.info(f"Attempting to load data from {full_path}")
                return pd.read_csv(str(full_path), index_col=0)
            except FileNotFoundError:
                # If that fails, try the relative fallback path
                fallback_path = self.root_dir / relative_fallback
                self.log.warning(
                    f"File not found at {full_path}, falling back to {fallback_path}"
                )
                return pd.read_csv(str(fallback_path), index_col=0)

        # Load test data with fallbacks
        try:
            if "SPLIT_DATA" in self.config:
                x_test_path = self.config["SPLIT_DATA"]["X_test"]
                y_test_path = self.config["SPLIT_DATA"]["y_test"]
                self.log.info(
                    f"Using test data paths from config: {x_test_path}, {y_test_path}"
                )
            else:
                x_test_path = "data/Test_Penguins_X.csv"
                y_test_path = "data/Test_Penguins_y.csv"
                self.log.info(
                    f"Using default test data paths: {x_test_path}, {y_test_path}"
                )

            self.X_test = safe_load_csv(x_test_path, "data/Test_Penguins_X.csv")
            self.y_test = safe_load_csv(y_test_path, "data/Test_Penguins_y.csv")
            self.log.info(
                f"Successfully loaded test data with shapes: X={self.X_test.shape}, y={self.y_test.shape}"
            )
        except Exception as e:
            self.log.error(f"Error loading test data: {e}")
            self.log.error(traceback.format_exc())
            raise

        self.log.info("Predictor is ready")

    def predict(self) -> bool:
        """
        Make predictions using the trained model.

        Returns:
            bool: True if operation is successful, False otherwise.
        """
        try:
            args = self.parser.parse_args()
            test_type = args.tests

            if not os.path.isfile(self.model_path):
                self.log.error(f"Model file not found at {self.model_path}")
                return False

            with open(self.model_path, "rb") as f:
                model = pickle.load(f)

            self.log.info(f"Model loaded from {self.model_path}")

            results_dir = str(self.root_dir / "results")
            os.makedirs(results_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                results_dir, f"prediction_results_{timestamp}.json"
            )

            if test_type == "smoke":
                sample_size = min(5, len(self.X_test))
                sample_indices = np.random.choice(
                    len(self.X_test), sample_size, replace=False
                )
                X_sample = self.X_test.iloc[sample_indices]
                y_sample = self.y_test.iloc[sample_indices]

                y_pred = model.predict(X_sample)

                results = {
                    "test_type": "smoke",
                    "timestamp": timestamp,
                    "predictions": [],
                }

                for i, (idx, row) in enumerate(X_sample.iterrows()):
                    pred_entry = {
                        "sample_id": int(idx),
                        "features": row.to_dict(),
                        "predicted_species": y_pred[i],
                        "actual_species": y_sample.iloc[i].values[0],
                    }
                    results["predictions"].append(pred_entry)

                with open(results_file, "w") as f:
                    json.dump(results, f, indent=4)

                self.log.info(f"Smoke test completed, results saved to {results_file}")

            elif test_type == "func":
                y_pred = model.predict(self.X_test)
                accuracy = sum(y_pred == self.y_test.values.ravel()) / len(y_pred)

                results = {
                    "test_type": "functional",
                    "timestamp": timestamp,
                    "accuracy": float(accuracy),
                    "num_samples": len(self.X_test),
                    "sample_predictions": [],
                }

                sample_size = min(10, len(self.X_test))
                for i in range(sample_size):
                    pred_entry = {
                        "sample_id": int(self.X_test.index[i]),
                        "predicted_species": y_pred[i],
                        "actual_species": self.y_test.iloc[i].values[0],
                    }
                    results["sample_predictions"].append(pred_entry)

                with open(results_file, "w") as f:
                    json.dump(results, f, indent=4)

                self.log.info(
                    f"Functional test completed with accuracy: {accuracy:.4f}"
                )
                self.log.info(f"Results saved to {results_file}")

            return True

        except Exception as e:
            self.log.error(f"Error in predict: {e}")
            self.log.error(traceback.format_exc())
            return False


if __name__ == "__main__":
    predictor = PenguinPredictor()
    predictor.predict()
