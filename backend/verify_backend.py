import requests
import os
import time
import json

BASE_URL = "http://localhost:8000"
LOG_FILE = "backend_verification.log"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

def test_login():
    log("\n--- Testing Login ---")
    payload = {"username": "admin", "password": "admin"}
    try:
        res = requests.post(f"{BASE_URL}/login", json=payload)
        log(f"Status: {res.status_code}")
        log(f"Response: {json.dumps(res.json())}")
    except Exception as e:
        log(f"Login Failed: {e}")

def test_upload_and_ingest():
    log("\n--- Testing Upload & Ingestion ---")
    # STRICT 5-COLUMN CHECK COMPLIANT CSV
    # Required: port_of_arr, port_of_dep, carrier_scac_code (4), master_carrier_scac_code (4), vessel_imo (7)
    csv_content = (
        "port_of_dep,port_of_arr,carrier_scac_code,master_carrier_scac_code,vessel_imo,ModeOfTransport,Actual_Duration_Hours\n"
        "USLAX,CNSHA,MSCI,MAEU,1234567,OCEAN,500\n"
        "NLROT,BEANR,CMAU,MSCU,7654321,OCEAN,120\n"
    )
    filename = "test_upload_valid.csv"
    with open(filename, "w") as f:
        f.write(csv_content)
    
    try:
        # 1. Upload
        files = {'file': (filename, open(filename, 'rb'), 'text/csv')}
        res = requests.post(f"{BASE_URL}/upload", files=files)
        log(f"Upload Status: {res.status_code}")
        data = res.json()
        log(f"Upload Response: {json.dumps(data)}")
        
        if res.status_code == 200 and data.get("can_ingest"):
            # 2. Ingest
            ingest_payload = {"filename": data["filename"]}
            res_ingest = requests.post(f"{BASE_URL}/ingest", json=ingest_payload)
            log(f"Ingest Status: {res_ingest.status_code}")
            log(f"Ingest Response: {json.dumps(res_ingest.json())}")
        else:
            log(f"Skipping ingestion. Can Ingest: {data.get('can_ingest')}")
            
    except Exception as e:
        log(f"Upload/Ingest Failed: {e}")
    finally:
        try:
            time.sleep(1)
            # if os.path.exists(filename):
            #    os.remove(filename) 
        except:
             pass

def test_predict():
    log("\n--- Testing Simulation ---")
    payload = {
        "PolCode": "USLAX",
        "PodCode": "CNSHA",
        "ModeOfTransport": "Ocean",
        "congestion_level": 10,
        "weather_severity": 20
    }
    try:
        res = requests.post(f"{BASE_URL}/simulate", json=payload)
        log(f"Predict Status: {res.status_code}")
        try:
            log(f"Predict Response: {json.dumps(res.json())}")
        except:
             log(f"Predict Response (Raw): {res.text}")
    except Exception as e:
        log(f"Predict Failed: {e}")

if __name__ == "__main__":
    test_login()
    test_upload_and_ingest()
    test_predict()
