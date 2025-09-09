import json
import requests
import sys
import time
from datetime import datetime


class ScenarioTester:
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.results = []
        
    def load_scenarios(self, scenario_file="scenario.json"):
        """Load test scenarios from JSON file."""
        try:
            with open(scenario_file, 'r') as f:
                data = json.load(f)
            return data.get('scenarios', [])
        except FileNotFoundError:
            print(f"Error: Scenario file {scenario_file} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {scenario_file}: {e}")
            return []
    
    def run_scenario(self, scenario):
        """Run a single test scenario."""
        print(f"\n--- Running: {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        
        try:
            url = f"{self.base_url}{scenario['endpoint']}"
            method = scenario['method']
            
            # Prepare request
            if method == 'GET':
                response = requests.get(url, timeout=10)
            elif method == 'POST':
                headers = {'Content-Type': 'application/json'}
                payload = scenario.get('payload', {})
                response = requests.post(url, json=payload, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check status code
            expected_status = scenario.get('expected_status', 200)
            status_match = response.status_code == expected_status
            
            # Check response content if specified
            content_match = True
            if 'expected_response' in scenario:
                try:
                    response_json = response.json()
                    expected = scenario['expected_response']
                    content_match = self._check_response_content(response_json, expected)
                except json.JSONDecodeError:
                    content_match = False
            
            # Record result
            result = {
                'scenario': scenario['name'],
                'status_code': response.status_code,
                'expected_status': expected_status,
                'status_match': status_match,
                'content_match': content_match,
                'success': status_match and content_match,
                'response': response.text[:500] if response.text else "No response body",
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            # Print result
            if result['success']:
                print(f"PASSED - Status: {response.status_code}")
            else:
                print(f"FAILED - Status: {response.status_code} (expected: {expected_status})")
                if not status_match:
                    print(f"   Status code mismatch")
                if not content_match:
                    print(f"   Response content mismatch")
                print(f"   Response: {response.text[:200]}...")
            
            return result['success']
            
        except requests.exceptions.RequestException as e:
            print(f"FAILED - Request error: {e}")
            result = {
                'scenario': scenario['name'],
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            return False
        except Exception as e:
            print(f"FAILED - Unexpected error: {e}")
            result = {
                'scenario': scenario['name'],
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            return False
    
    def _check_response_content(self, response_json, expected):
        """Check if response content matches expected values."""
        try:
            for key, expected_value in expected.items():
                if key not in response_json:
                    return False
                if response_json[key] != expected_value:
                    return False
            return True
        except Exception:
            return False
    
    def run_all_scenarios(self, scenario_file="scenario.json"):
        """Run all scenarios from the scenario file."""
        print("=" * 60)
        print("SCENARIO-BASED TESTING")
        print("=" * 60)
        print(f"Base URL: {self.base_url}")
        print(f"Scenario file: {scenario_file}")
        print(f"Started at: {datetime.now().isoformat()}")
        
        scenarios = self.load_scenarios(scenario_file)
        if not scenarios:
            print("No scenarios found to run")
            return False
        
        print(f"Found {len(scenarios)} scenarios to run")
        
        passed = 0
        failed = 0
        
        for scenario in scenarios:
            if self.run_scenario(scenario):
                passed += 1
            else:
                failed += 1
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total scenarios: {len(scenarios)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success rate: {(passed/len(scenarios)*100):.1f}%")
        
        if failed == 0:
            print("\nALL SCENARIOS PASSED!")
            return True
        else:
            print(f"\n{failed} SCENARIOS FAILED")
            return False
    
    def generate_report(self, filename="scenario_test_report.json"):
        """Generate a detailed test report."""
        report = {
            'test_run': {
                'timestamp': datetime.now().isoformat(),
                'base_url': self.base_url,
                'total_scenarios': len(self.results),
                'passed': sum(1 for r in self.results if r['success']),
                'failed': sum(1 for r in self.results if not r['success'])
            },
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {filename}")


def main():
    """Main function to run scenario tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run scenario-based tests for Penguin Classifier API')
    parser.add_argument('--url', default='http://localhost:5001', 
                       help='Base URL for the API (default: http://localhost:5001)')
    parser.add_argument('--scenario-file', default='scenario.json',
                       help='Path to scenario JSON file (default: scenario.json)')
    parser.add_argument('--report', default='scenario_test_report.json',
                       help='Output file for test report (default: scenario_test_report.json)')
    
    args = parser.parse_args()
    
    # Create tester and run scenarios
    tester = ScenarioTester(args.url)
    
    # Wait a moment for API to be ready
    print("Waiting for API to be ready...")
    time.sleep(2)
    
    # Run all scenarios
    success = tester.run_all_scenarios(args.scenario_file)
    
    # Generate report
    tester.generate_report(args.report)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
