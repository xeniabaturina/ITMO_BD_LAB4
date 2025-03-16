import requests
import json
import sys
import os


def run_scenario_tests(scenario_file):
    # Load scenarios from file
    try:
        with open(scenario_file, "r") as f:
            test_suite = json.load(f)
    except Exception as e:
        print(f"Error loading scenario file: {e}")
        return False

    print(f"Running test suite: {test_suite['name']}")
    print(f"Description: {test_suite['description']}")
    print(f"Version: {test_suite['version']}")
    print(f"Total scenarios: {len(test_suite['scenarios'])}")

    # Track test results
    passed = 0
    failed = 0

    # Run each scenario
    for scenario in test_suite["scenarios"]:
        print(f"\nRunning scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")

        # Prepare request
        url = f"http://localhost:5000{scenario['endpoint']}"
        method = scenario["method"]

        try:
            # Make request based on method
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                response = requests.post(url, json=scenario.get("payload", {}))
            else:
                print(f"Unsupported method: {method}")
                failed += 1
                continue

            # Check status code
            expected_status = scenario["expected_status"]
            if response.status_code != expected_status:
                print(
                    f"❌ Status code mismatch: expected {expected_status}, got {response.status_code}"
                )
                failed += 1
                continue

            # For successful responses, check expected response content if provided
            if "expected_response" in scenario and response.status_code == 200:
                try:
                    response_data = response.json()
                    expected_data = scenario["expected_response"]

                    # Check each expected field
                    all_matched = True
                    for key, expected_value in expected_data.items():
                        if key not in response_data:
                            print(f"❌ Missing field in response: {key}")
                            all_matched = False
                        elif response_data[key] != expected_value:
                            print(
                                f"❌ Value mismatch for {key}: expected {expected_value}, got {response_data[key]}"
                            )
                            all_matched = False

                    if not all_matched:
                        failed += 1
                        continue
                except Exception as e:
                    print(f"❌ Error parsing response: {e}")
                    failed += 1
                    continue

            # If we got here, the test passed
            print("✅ Scenario passed")
            passed += 1

        except Exception as e:
            print(f"❌ Error executing scenario: {e}")
            failed += 1

    # Print summary
    print("\nTest Summary:")
    print(f"Total scenarios: {len(test_suite['scenarios'])}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    # Return True if all tests passed
    return failed == 0


if __name__ == "__main__":
    scenario_file = "scenario.json"
    if not os.path.exists(scenario_file):
        print(f"Error: Scenario file '{scenario_file}' not found")
        sys.exit(1)

    success = run_scenario_tests(scenario_file)
    if success:
        print("All scenarios passed!")
        sys.exit(0)
    else:
        print("Some scenarios failed")
        sys.exit(1)
