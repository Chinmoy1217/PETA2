import requests
import pandas as pd
import io

BASE_URL = "http://127.0.0.1:8000"

def test_simulate():
    print("\n[TEST] Testing /simulate endpoint...")
    payload = {
        "PolCode": "CNSHG",
        "PodCode": "USLAX",
        "ModeOfTransport": "OCEAN",
        "congestion_level": 50,
        "weather_severity": 20
    }
    try:
        resp = requests.post(f"{BASE_URL}/simulate", json=payload)
        if resp.status_code == 200:
            print("✅ /simulate Success")
            print(f"   -> AI Prediction: {resp.json().get('prediction_hours')} hours")
            print(f"   -> Risk Score: {resp.json().get('risk_score')}")
        else:
            print(f"❌ /simulate Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

def test_upload():
    print("\n[TEST] Testing /predict (CSV Upload)...")
    # Create mock CSV
    csv_content = "PolCode,PodCode,ModeOfTransport,Actual_Duration_Hours\nCNSHG,USLAX,OCEAN,0\nINBOM,AEDXB,AIR,0"
    files = {'file': ('test.csv', csv_content, 'text/csv')}
    
    try:
        resp = requests.post(f"{BASE_URL}/predict", files=files)
        if resp.status_code == 200:
            data = resp.json()
            print("✅ Upload Success")
            print(f"   -> Processed: {data.get('count')} records")
            print(f"   -> Summary: {data.get('ai_summary')}")
        else:
            print(f"❌ Upload Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_simulate()
    test_upload()
