import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_simulation():
    print("\n--- Testing Single Simulation (Hub Logic) ---")
    url = f"{BASE_URL}/simulate"
    
    # Test Case: Shanghai to Rotterdam (Should trigger Transshipment via SGSIN/EGSUEZ)
    payload = {
        "PolCode": "CNSHA",
        "PodCode": "NLROT",
        "ModeOfTransport": "OCEAN",
        "congestion_level": 50,
        "weather_severity": 20
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            print("âœ… Simulation Success")
            print(f"Prediction: {data['prediction_hours']} hours ({data['prediction_days']} days)")
            print(f"Confidence: {data.get('confidence_score')}%")
            print(f"Risk Score: {data.get('risk_score')}")
            
            # Check for Transshipment
            route = data.get('route_details', {})
            print(f"Via Port: {route.get('via_port')}")
            
            if route.get('via_port') != 'DIRECT':
                print("âœ… Start/End Continent difference correctly triggered Hub logic.")
            
            # Check Explanation
            print(f"\nAI Explanation:\n{data.get('ai_explanation')}")
        else:
            print(f"âŒ Failed: {response.text}")
    except Exception as e:
        print(f"Connection Error (Is server running?): {e}")

def test_predict_validation():
    print("\n--- Testing Geo-Validation (Island Check) ---")
    url = f"{BASE_URL}/predict"
    
    # Test Case: Trucking to Island (Sri Lanka) -> Should Fail
    data = {
        "pol": "INMAA", # Chennai
        "pod": "LKCMB", # Colombo (Island)
        "mode": "TRUCK"
    }
    
    try:
        response = requests.post(url, data=data)
        if response.status_code == 400:
            print(f"âœ… Validation Working: {response.json()['detail']}")
        elif response.status_code == 200:
            print("âŒ Validation Failed (Request succeeded unexpectedly)")
        else:
            print(f"Status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_simulation()
    test_predict_validation()
