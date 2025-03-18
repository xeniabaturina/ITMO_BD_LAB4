import configparser
import os
import pickle
import pandas as pd
import traceback
import json
import datetime
from flask import Flask, request, jsonify, g
from pathlib import Path

from .logger import Logger
from .database import init_db, get_db, save_prediction, PredictionResult

SHOW_LOG = True
app = Flask(__name__)
logger = Logger(SHOW_LOG)
log = logger.get_logger(__name__)

# Initialize database on first request
db_initialized = False


def get_db_connection():
    global db_initialized
    if not db_initialized:
        init_db()
        db_initialized = True
    return next(get_db())


@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


class ModelService:
    """
    Class for serving the penguin classification model via an API.
    """

    def __init__(self):
        self.config = configparser.ConfigParser()
        # Get the project root directory (assuming src is one level below root)
        self.root_dir = Path(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config_path = self.root_dir / "config.ini"
        self.config.read(str(config_path))

        self.project_path = str(self.root_dir / "experiments")
        self.model_path = str(Path(self.project_path) / "random_forest.sav")

        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            log.info(f"Model loaded from {self.model_path}")
        except FileNotFoundError:
            log.error(f"Model file not found at {self.model_path}")
            self.model = None
        except Exception as e:
            log.error(f"Error loading model: {e}")
            log.error(traceback.format_exc())
            self.model = None

    def predict(self, data):
        """
        Make predictions with the loaded model.

        Args:
            data (dict): Input data for prediction.

        Returns:
            dict: Prediction results.
        """
        try:
            # Validate input data types
            self._validate_input_data(data)

            # Prepare input data
            input_df = pd.DataFrame([data])

            # Make prediction
            species = self.model.predict(input_df)[0]
            probabilities = self.model.predict_proba(input_df)[0]
            class_labels = self.model.classes_

            prob_dict = {
                label: float(prob) for label, prob in zip(class_labels, probabilities)
            }

            # Get the confidence score for the predicted species
            confidence = prob_dict[species]

            # Store prediction in database
            try:
                db = get_db_connection()
                save_prediction(
                    db=db,
                    culmen_length_mm=data["bill_length_mm"],
                    culmen_depth_mm=data["bill_depth_mm"],
                    flipper_length_mm=data["flipper_length_mm"],
                    body_mass_g=data["body_mass_g"],
                    predicted_species=species,
                    confidence=confidence,
                )
            except Exception as e:
                log.error(f"Error saving prediction to database: {e}")
                # Continue with prediction even if database save fails

            return {
                "success": True,
                "predicted_species": species,
                "probabilities": prob_dict,
            }

        except ValueError as e:
            log.error(f"Input validation error: {e}")
            return {"success": False, "error": str(e), "error_type": "validation_error"}
        except Exception as e:
            log.error(f"Error during prediction: {e}")
            log.error(traceback.format_exc())
            return {"success": False, "error": str(e), "error_type": "prediction_error"}

    def _validate_input_data(self, data):
        """
        Validate input data types and ranges.

        Args:
            data (dict): Input data for prediction.

        Raises:
            ValueError: If input data is invalid.
        """
        # Check island
        if not isinstance(data.get("island"), str):
            raise ValueError("Island must be a string")

        # Check numeric fields
        numeric_fields = {
            "bill_length_mm": (10.0, 60.0),
            "bill_depth_mm": (10.0, 30.0),
            "flipper_length_mm": (150.0, 250.0),
            "body_mass_g": (2500.0, 6500.0),
        }

        for field, (min_val, max_val) in numeric_fields.items():
            value = data.get(field)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be a number")
            if value < min_val or value > max_val:
                raise ValueError(f"{field} must be between {min_val} and {max_val}")

        # Check sex
        if not isinstance(data.get("sex"), str):
            raise ValueError("Sex must be a string")

        # Normalize sex field
        data["sex"] = data["sex"].upper()
        if data["sex"] not in ["MALE", "FEMALE"]:
            raise ValueError("Sex must be either 'MALE' or 'FEMALE'")


def log_request(request_data, response_data, endpoint):
    """Log API requests and responses to a file."""
    # Get the project root directory (assuming src is one level below root)
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = str(root_dir / "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir, f"api_requests_{datetime.datetime.now().strftime('%Y%m%d')}.log"
    )

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "endpoint": endpoint,
        "request": request_data,
        "response": response_data,
        "status_code": 200 if response_data.get("success", False) else 400,
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


model_service = ModelService()


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    Returns the status of the API and whether the model is loaded.
    """
    try:
        # Check if database is accessible
        db_status = "connected"
        try:
            get_db_connection()
        except Exception as e:
            db_status = f"error: {str(e)}"

        # Check if model is loaded
        model_loaded = model_service.model is not None

        # Return health status
        return jsonify(
            {
                "status": "healthy",
                "model_loaded": model_loaded,
                "database": db_status,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )
    except Exception as e:
        log.error(f"Health check failed: {str(e)}")
        return jsonify(
            {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
            }
        ), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.

    Expected JSON input:
    {
        "island": "Torgersen",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "sex": "male"
    }
    """
    request_data = request.get_json()

    # Initial validation
    if model_service.model is None:
        response = {
            "success": False,
            "error": "Model not loaded",
            "error_type": "server_error",
        }
        log_request(request_data, response, "predict")
        return jsonify(response), 500

    try:
        if not request_data:
            response = {
                "success": False,
                "error": "No input data provided",
                "error_type": "validation_error",
            }
            log_request(request_data, response, "predict")
            return jsonify(response), 400

        required_fields = [
            "island",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ]
        missing_fields = [
            field for field in required_fields if field not in request_data
        ]

        if missing_fields:
            response = {
                "success": False,
                "error": f"Missing required fields: {missing_fields}",
                "error_type": "validation_error",
            }
            log_request(request_data, response, "predict")
            return jsonify(response), 400

        result = model_service.predict(request_data)
        log_request(request_data, result, "predict")

        if result["success"]:
            return jsonify(result), 200
        else:
            error_code = 400 if result.get("error_type") == "validation_error" else 500
            return jsonify(result), error_code

    except Exception as e:
        log.error(f"Error in predict endpoint: {e}")
        log.error(traceback.format_exc())
        response = {"success": False, "error": str(e), "error_type": "server_error"}
        log_request(request_data, response, "predict")
        return jsonify(response), 500


@app.route("/predictions", methods=["GET"])
def get_predictions():
    """
    Retrieve prediction history from the database.
    Optional query parameters:
    - limit: maximum number of predictions to return (default: 100)
    - offset: number of predictions to skip (default: 0)
    """
    try:
        limit = min(int(request.args.get("limit", 100)), 1000)
        offset = int(request.args.get("offset", 0))

        log.info(f"Retrieving predictions with limit={limit}, offset={offset}")

        try:
            log.info("Getting database connection...")
            db = get_db_connection()
            log.info("Database connection established")

            # Get predictions ordered by timestamp
            log.info("Querying database...")

            # Safety check - use a very small limit for testing
            if os.environ.get("TESTING") == "1":
                log.info("In testing mode, using small limit")
                limit = min(limit, 5)  # Use a small limit in testing mode

            results = (
                db.query(PredictionResult)
                .order_by(PredictionResult.timestamp.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            log.info(f"Retrieved {len(results)} predictions from database")

            predictions = []
            log.info("Processing results...")
            for result in results:
                predictions.append(
                    {
                        "id": result.id,
                        "timestamp": result.timestamp.isoformat(),
                        "culmen_length_mm": result.culmen_length_mm,
                        "culmen_depth_mm": result.culmen_depth_mm,
                        "flipper_length_mm": result.flipper_length_mm,
                        "body_mass_g": result.body_mass_g,
                        "predicted_species": result.predicted_species,
                        "confidence": result.confidence,
                    }
                )

            log.info("Returning response...")
            return jsonify(
                {
                    "success": True,
                    "predictions": predictions,
                    "count": len(predictions),
                    "offset": offset,
                    "limit": limit,
                }
            )
        except Exception as e:
            log.error(f"Database error: {e}")
            log.error(traceback.format_exc())
            return jsonify(
                {
                    "success": False,
                    "error": f"Database error: {str(e)}",
                    "error_type": "database_error",
                }
            ), 500

    except Exception as e:
        log.error(f"Error retrieving predictions: {e}")
        log.error(traceback.format_exc())
        return jsonify(
            {
                "success": False,
                "error": f"Error retrieving predictions: {str(e)}",
                "error_type": "server_error",
            }
        ), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
