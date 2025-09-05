"""
Kafka Producer for Penguin Classifier
Sends prediction results to Kafka topic for processing by consumers.
"""

import os
import json
import time
import traceback
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError

from .logger import Logger
from .database import init_db, get_db, PredictionResult
from .secrets_manager import get_secrets_manager

SHOW_LOG = True
logger = Logger(SHOW_LOG)
log = logger.get_logger(__name__)


class KafkaProducerService:
    """
    Service for producing messages to Kafka topics.
    """

    def __init__(self):
        # Get Kafka configuration from secrets manager or environment
        try:
            secrets_manager = get_secrets_manager()
            kafka_config = secrets_manager.get_kafka_config()
            self.bootstrap_servers = kafka_config.get('bootstrap_servers', 'localhost:9092')
            self.topic_name = kafka_config.get('topic_name', 'penguin-predictions')
        except Exception as e:
            log.warning(f"Could not get Kafka config from secrets manager: {e}")
            log.warning("Falling back to environment variables")
            self.bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
            self.topic_name = os.environ.get('KAFKA_TOPIC_NAME', 'penguin-predictions')
        
        self.producer = None
        self._initialize_producer()

    def _initialize_producer(self):
        """Initialize Kafka producer with proper configuration."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retries=3,
                retry_backoff_ms=1000,
                request_timeout_ms=30000,
                api_version=(0, 10, 1)
            )
            log.info(f"Kafka producer initialized with servers: {self.bootstrap_servers}")
        except Exception as e:
            log.error(f"Failed to initialize Kafka producer: {e}")
            log.error(traceback.format_exc())
            self.producer = None

    def send_prediction_result(self, prediction_data):
        """
        Send prediction result to Kafka topic.
        
        Args:
            prediction_data (dict): Prediction data to send
        """
        if not self.producer:
            log.error("Kafka producer not initialized")
            return False

        try:
            # Add timestamp to the message
            message = {
                'timestamp': datetime.now().isoformat(),
                'prediction_data': prediction_data,
                'source': 'penguin-classifier'
            }

            # Send message to Kafka
            future = self.producer.send(
                self.topic_name,
                key=str(prediction_data.get('id', 'unknown')),
                value=message
            )

            # Wait for the message to be sent
            record_metadata = future.get(timeout=10)
            log.info(f"Message sent to topic {record_metadata.topic} "
                    f"partition {record_metadata.partition} "
                    f"offset {record_metadata.offset}")
            return True

        except KafkaError as e:
            log.error(f"Kafka error while sending message: {e}")
            return False
        except Exception as e:
            log.error(f"Error sending message to Kafka: {e}")
            log.error(traceback.format_exc())
            return False

    def close(self):
        """Close the Kafka producer."""
        if self.producer:
            self.producer.close()
            log.info("Kafka producer closed")


def get_db_connection():
    """Get database connection."""
    try:
        init_db()
        return next(get_db())
    except Exception as e:
        log.error(f"Error getting database connection: {e}")
        return None


def main():
    """Main function to run the Kafka producer service."""
    log.info("Starting Kafka Producer Service...")
    
    # Initialize Kafka producer
    kafka_producer = KafkaProducerService()
    
    if not kafka_producer.producer:
        log.error("Failed to initialize Kafka producer. Exiting.")
        return

    # Initialize database
    db = get_db_connection()
    if not db:
        log.error("Failed to connect to database. Exiting.")
        return

    log.info("Kafka Producer Service started successfully")
    log.info("Monitoring database for new predictions...")

    try:
        last_processed_id = 0
        
        while True:
            try:
                # Query for new predictions since last processed
                new_predictions = (
                    db.query(PredictionResult)
                    .filter(PredictionResult.id > last_processed_id)
                    .order_by(PredictionResult.id.asc())
                    .limit(10)
                    .all()
                )

                for prediction in new_predictions:
                    # Convert prediction to dictionary
                    prediction_data = {
                        'id': prediction.id,
                        'timestamp': prediction.timestamp.isoformat(),
                        'culmen_length_mm': prediction.culmen_length_mm,
                        'culmen_depth_mm': prediction.culmen_depth_mm,
                        'flipper_length_mm': prediction.flipper_length_mm,
                        'body_mass_g': prediction.body_mass_g,
                        'predicted_species': prediction.predicted_species,
                        'confidence': prediction.confidence
                    }

                    # Send to Kafka
                    success = kafka_producer.send_prediction_result(prediction_data)
                    if success:
                        last_processed_id = prediction.id
                        log.info(f"Sent prediction {prediction.id} to Kafka")
                    else:
                        log.error(f"Failed to send prediction {prediction.id} to Kafka")

                # Sleep for a short interval before checking again
                time.sleep(5)

            except Exception as e:
                log.error(f"Error in main loop: {e}")
                log.error(traceback.format_exc())
                time.sleep(10)  # Wait longer on error

    except KeyboardInterrupt:
        log.info("Received interrupt signal. Shutting down...")
    except Exception as e:
        log.error(f"Unexpected error in main: {e}")
        log.error(traceback.format_exc())
    finally:
        kafka_producer.close()
        log.info("Kafka Producer Service stopped")


if __name__ == "__main__":
    main()
