import requests
import os

BASE_URL = "http://localhost:8000"

def test_login():
    print("\n--- Testing Login ---")
    payload = {"username": "admin", "password": "admin"}
    try:
        res = requests.post(f"{BASE_URL}/login", json=payload)
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
    except Exception as e:
        print(f"Login Failed: {e}")

def test_upload_and_ingest():
    print("\n--- Testing Upload & Ingestion ---")
    # Create valid CSV
    csv_content = "PORT_OF_DEP,PORT_OF_ARR,MODE_OF_TRANSPORT,Actual_Duration_Hours\nUSLAX,CNSHA,OCEAN,500\n"
    with open("test_upload.csv", "w") as f:
        f.write(csv_content)
    
    # 1. Upload
    try:
        files = {'file': ('test_upload.csv', open('test_upload.csv', 'rb'), 'text/csv')}
        res = requests.post(f"{BASE_URL}/upload", files=files)
        print(f"Upload Status: {res.status_code}")
        data = res.json()
        print(f"Upload Response: {data}")
        
        if res.status_code == 200 and data.get("can_ingest"):
            # 2. Ingest
            ingest_payload = {"filename": data["filename"]}
            res_ingest = requests.post(f"{BASE_URL}/ingest", json=ingest_payload)
            print(f"Ingest Status: {res_ingest.status_code}")
            print(f"Ingest Response: {res_ingest.json()}")
        else:
            print("Skipping ingestion due to upload failure or quality check.")
            
    except Exception as e:
        print(f"Upload/Ingest Failed: {e}")
    finally:
        if os.path.exists("test_upload.csv"):
            os.remove("test_upload.csv")

def test_predict():
    print("\n--- Testing Single Prediction ---")
    payload = {
        "PolCode": "USLAX",
        "PodCode": "CNSHA",
        "ModeOfTransport": "Ocean",
        "congestion_level": 10,
        "weather_severity": 20
    }
    try:
        res = requests.post(f"{BASE_URL}/simulate", json=payload)
        print(f"Predict Status: {res.status_code}")
        print(f"Predict Response: {res.json()}")
    except Exception as e:
        print(f"Predict Failed: {e}")

if __name__ == "__main__":
    test_login()
    test_upload_and_ingest()
    test_predict()
