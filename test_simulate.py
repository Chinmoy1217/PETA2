import requests
import json

try:
    url = "http://localhost:8000/simulate"
    payload = {
        "PolCode": "USLAX",
        "PodCode": "CNSHA",
        "ModeOfTransport": "Ocean"
    }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
