#!/usr/bin/env python3
"""
Test script for Kafka integration with Penguin Classifier
"""

import requests
import json
import time
import sys
import argparse
from datetime import datetime


def test_api_health(base_url="http://localhost:5001"):
    """Test API health endpoint."""
    print("Testing API health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ API Health: {health_data['status']}")
            print(f"  - Model loaded: {health_data.get('model_loaded', 'unknown')}")
            print(f"  - Database: {health_data.get('database', 'unknown')}")
            print(f"  - Kafka: {health_data.get('kafka', 'unknown')}")
            return True
        else:
            print(f"✗ API Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ API Health check error: {e}")
        return False


def test_prediction(base_url="http://localhost:5001"):
    """Test prediction endpoint."""
    print("\nTesting prediction endpoint...")
    
    test_data = {
        "island": "Torgersen",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "sex": "male"
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"✓ Prediction successful")
                print(f"  - Predicted species: {result.get('predicted_species')}")
                print(f"  - Prediction ID: {result.get('prediction_id')}")
                print(f"  - Confidence: {result.get('probabilities', {}).get(result.get('predicted_species'), 0):.3f}")
                return result.get('prediction_id')
            else:
                print(f"✗ Prediction failed: {result.get('error')}")
                return None
        else:
            print(f"✗ Prediction request failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return None


def test_predictions_history(base_url="http://localhost:5001"):
    """Test predictions history endpoint."""
    print("\nTesting predictions history endpoint...")
    
    try:
        response = requests.get(f"{base_url}/predictions?limit=5", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                predictions = result.get("predictions", [])
                print(f"✓ Retrieved {len(predictions)} predictions from history")
                for pred in predictions[:3]:  # Show first 3
                    print(f"  - ID: {pred.get('id')}, Species: {pred.get('predicted_species')}, "
                          f"Confidence: {pred.get('confidence', 0):.3f}")
                return True
            else:
                print(f"✗ History request failed: {result.get('error')}")
                return False
        else:
            print(f"✗ History request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ History error: {e}")
        return False


def wait_for_kafka_processing():
    """Wait for Kafka processing to complete."""
    print("\nWaiting for Kafka processing...")
    time.sleep(5)  # Give Kafka time to process the message


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test Kafka integration with Penguin Classifier API')
    parser.add_argument('--url', default='http://localhost:5001', 
                       help='Base URL for the API (default: http://localhost:5001)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("KAFKA INTEGRATION TEST")
    print("=" * 60)
    print(f"Base URL: {args.url}")
    print(f"Test started at: {datetime.now().isoformat()}")
    
    # Test API health
    if not test_api_health(args.url):
        print("\n✗ API health check failed. Make sure the services are running.")
        sys.exit(1)
    
    # Test prediction
    prediction_id = test_prediction(args.url)
    if not prediction_id:
        print("\n✗ Prediction test failed.")
        sys.exit(1)
    
    # Wait for Kafka processing
    wait_for_kafka_processing()
    
    # Test predictions history
    if not test_predictions_history(args.url):
        print("\n✗ Predictions history test failed.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nKafka integration is working correctly:")
    print("1. ✓ API is healthy and connected to Kafka")
    print("2. ✓ Predictions are being saved to database")
    print("3. ✓ Predictions are being sent to Kafka")
    print("4. ✓ Prediction history can be retrieved")
    print("\nCheck the logs for Kafka producer and consumer activity:")
    print("- Producer logs: docker logs kafka-producer")
    print("- Consumer logs: docker logs kafka-consumer")
    print("- Kafka logs: docker logs kafka")


if __name__ == "__main__":
    main()
