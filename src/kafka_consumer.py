"""
Kafka Consumer for Penguin Classifier
Consumes prediction results from Kafka topic and processes them.
"""

import os
import json
import time
import traceback
import configparser
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from .logger import Logger
from .database import init_db, get_db, PredictionResult
from .secrets_manager import get_secrets_manager

SHOW_LOG = True
logger = Logger(SHOW_LOG)
log = logger.get_logger(__name__)


class KafkaConsumerService:
    """
    Service for consuming messages from Kafka topics.
    """

    def __init__(self):
        # Load Kafka configuration from config file
        self.config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini')
        self.config.read(config_path)
        
        # Get Kafka configuration from config file with environment variable overrides
        kafka_config = self.config['KAFKA']
        self.bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', kafka_config.get('bootstrap_servers'))
        self.topic_name = os.environ.get('KAFKA_TOPIC_NAME', kafka_config.get('topic_name'))
        self.group_id = os.environ.get('KAFKA_GROUP_ID', kafka_config.get('group_id'))
        
        # Additional consumer configuration from config file
        self.auto_offset_reset = kafka_config.get('auto_offset_reset', 'earliest')
        self.enable_auto_commit = kafka_config.getboolean('enable_auto_commit', True)
        self.auto_commit_interval_ms = kafka_config.getint('auto_commit_interval_ms', 1000)
        self.session_timeout_ms = kafka_config.getint('session_timeout_ms', 30000)
        self.heartbeat_interval_ms = kafka_config.getint('heartbeat_interval_ms', 10000)
        
        # Validate required configuration
        if not self.bootstrap_servers or not self.topic_name or not self.group_id:
            raise ValueError("Missing required Kafka configuration")
            
        log.info(f"Kafka configuration loaded: servers={self.bootstrap_servers}, topic={self.topic_name}, group={self.group_id}")
        
        self.consumer = None
        self._initialize_consumer()

    def _initialize_consumer(self):
        """Initialize Kafka consumer with proper configuration."""
        try:
            self.consumer = KafkaConsumer(
                self.topic_name,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=self.enable_auto_commit,
                auto_commit_interval_ms=self.auto_commit_interval_ms,
                session_timeout_ms=self.session_timeout_ms,
                heartbeat_interval_ms=self.heartbeat_interval_ms,
                api_version=(0, 10, 1)
            )
            log.info(f"Kafka consumer initialized with servers: {self.bootstrap_servers}")
            log.info(f"Subscribed to topic: {self.topic_name}")
            log.info(f"Consumer group: {self.group_id}")
        except Exception as e:
            log.error(f"Failed to initialize Kafka consumer: {e}")
            log.error(traceback.format_exc())
            self.consumer = None

    def process_message(self, message):
        """
        Process a single message from Kafka.
        
        Args:
            message: Kafka message object
        """
        try:
            # Extract message data
            key = message.key
            value = message.value
            topic = message.topic
            partition = message.partition
            offset = message.offset

            log.info(f"Processing message - Topic: {topic}, Partition: {partition}, "
                    f"Offset: {offset}, Key: {key}")

            # Validate message structure
            if not isinstance(value, dict):
                log.error(f"Invalid message format: {value}")
                return False

            # Extract prediction data
            prediction_data = value.get('prediction_data', {})
            message_timestamp = value.get('timestamp')
            source = value.get('source', 'unknown')

            log.info(f"Received prediction from {source} at {message_timestamp}")
            log.info(f"Prediction data: {prediction_data}")

            # Process the prediction (example: log, store in secondary database, etc.)
            self._process_prediction(prediction_data, message_timestamp)

            return True

        except Exception as e:
            log.error(f"Error processing message: {e}")
            log.error(traceback.format_exc())
            return False

    def _process_prediction(self, prediction_data, message_timestamp):
        """
        Process prediction data (example implementation).
        
        Args:
            prediction_data (dict): Prediction data
            message_timestamp (str): Message timestamp
        """
        try:
            # Example processing: Log prediction details
            prediction_id = prediction_data.get('id')
            species = prediction_data.get('predicted_species')
            confidence = prediction_data.get('confidence')
            
            log.info(f"Processing prediction {prediction_id}: "
                    f"Species={species}, Confidence={confidence:.3f}")

            # Example: Store processed prediction in a separate table or file
            self._store_processed_prediction(prediction_data, message_timestamp)

            # Example: Trigger additional processing based on prediction
            self._trigger_additional_processing(prediction_data)

        except Exception as e:
            log.error(f"Error in prediction processing: {e}")
            log.error(traceback.format_exc())

    def _store_processed_prediction(self, prediction_data, message_timestamp):
        """
        Store processed prediction (example implementation).
        
        Args:
            prediction_data (dict): Prediction data
            message_timestamp (str): Message timestamp
        """
        try:
            # Example: Write to a log file or secondary database
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            processed_log_file = os.path.join(
                log_dir, 
                f"processed_predictions_{datetime.now().strftime('%Y%m%d')}.log"
            )
            
            log_entry = {
                'processed_at': datetime.now().isoformat(),
                'message_timestamp': message_timestamp,
                'prediction_data': prediction_data
            }
            
            with open(processed_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
            log.info(f"Stored processed prediction {prediction_data.get('id')} in log file")
            
        except Exception as e:
            log.error(f"Error storing processed prediction: {e}")

    def _trigger_additional_processing(self, prediction_data):
        """
        Trigger additional processing based on prediction (example implementation).
        
        Args:
            prediction_data (dict): Prediction data
        """
        try:
            species = prediction_data.get('predicted_species')
            confidence = prediction_data.get('confidence', 0)
            
            # Example: Alert if confidence is low
            if confidence < 0.7:
                log.warning(f"Low confidence prediction detected: "
                           f"Species={species}, Confidence={confidence:.3f}")
            
            # Example: Count predictions by species
            self._update_species_statistics(species)
            
        except Exception as e:
            log.error(f"Error in additional processing: {e}")

    def _update_species_statistics(self, species):
        """
        Update species statistics (example implementation).
        
        Args:
            species (str): Predicted species
        """
        try:
            # Example: Update statistics in a file
            stats_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'logs', 
                'species_statistics.json'
            )
            
            # Load existing statistics
            stats = {}
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
            
            # Update statistics
            stats[species] = stats.get(species, 0) + 1
            stats['last_updated'] = datetime.now().isoformat()
            
            # Save updated statistics
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
            log.info(f"Updated statistics for species: {species}")
            
        except Exception as e:
            log.error(f"Error updating species statistics: {e}")

    def start_consuming(self):
        """Start consuming messages from Kafka."""
        if not self.consumer:
            log.error("Kafka consumer not initialized")
            return

        log.info("Starting to consume messages from Kafka...")
        
        try:
            for message in self.consumer:
                success = self.process_message(message)
                if success:
                    log.info("Message processed successfully")
                else:
                    log.error("Failed to process message")
                    
        except KafkaError as e:
            log.error(f"Kafka error while consuming: {e}")
        except Exception as e:
            log.error(f"Error while consuming messages: {e}")
            log.error(traceback.format_exc())

    def close(self):
        """Close the Kafka consumer."""
        if self.consumer:
            self.consumer.close()
            log.info("Kafka consumer closed")


def get_db_connection():
    """Get database connection."""
    try:
        init_db()
        return next(get_db())
    except Exception as e:
        log.error(f"Error getting database connection: {e}")
        return None


def main():
    """Main function to run the Kafka consumer service."""
    log.info("Starting Kafka Consumer Service...")
    
    # Initialize Kafka consumer
    kafka_consumer = KafkaConsumerService()
    
    if not kafka_consumer.consumer:
        log.error("Failed to initialize Kafka consumer. Exiting.")
        return

    # Initialize database (for potential future use)
    db = get_db_connection()
    if not db:
        log.error("Failed to connect to database. Exiting.")
        return

    log.info("Kafka Consumer Service started successfully")
    log.info("Waiting for messages from Kafka...")

    try:
        kafka_consumer.start_consuming()
    except KeyboardInterrupt:
        log.info("Received interrupt signal. Shutting down...")
    except Exception as e:
        log.error(f"Unexpected error in main: {e}")
        log.error(traceback.format_exc())
    finally:
        kafka_consumer.close()
        log.info("Kafka Consumer Service stopped")


if __name__ == "__main__":
    main()
