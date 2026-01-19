import requests
import snowflake.connector
import pandas as pd
import json
import os
from backend.config import SNOWFLAKE_CONFIG

API_URL = "http://127.0.0.1:8000"

def get_snowflake_conn():
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        return conn
    except Exception as e:
        print(f"Snowflake Connection Failed: {e}")
        return None

def verify_snowflake():
    print("--- Verifying Snowflake ---")
    conn = get_snowflake_conn()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Check Tables
        cursor.execute("SHOW TABLES LIKE 'FACT_TRIP'")
        if not cursor.fetchone():
            print("❌ Table FACT_TRIP not found!")
        else:
            print("✅ Table FACT_TRIP found.")

        cursor.execute("SHOW TABLES LIKE 'MODEL_ACCURACY_HISTORY'")
        if not cursor.fetchone():
            print("❌ Table MODEL_ACCURACY_HISTORY not found!")
        else:
            print("✅ Table MODEL_ACCURACY_HISTORY found.")
            
        # Check Data Count
        cursor.execute("SELECT COUNT(*) FROM DT_INGESTION.FACT_TRIP")
        count = cursor.fetchone()[0]
        print(f"✅ FACT_TRIP Row Count: {count}")
        
        return True
    except Exception as e:
        print(f"❌ Snowflake Verification Failed: {e}")
        return False
    finally:
        conn.close()

def verify_backend():
    print("\n--- Verifying Backend ---")
    
    # 1. Metrics Endpoint
    try:
        res = requests.get(f"{API_URL}/metrics")
        if res.status_code == 200:
            data = res.json()
            print("✅ /metrics Endpoint is reachable.")
            print(f"   Accuracy: {data.get('accuracy')}%")
            print(f"   Total Shipments: {data.get('total_shipments')}")
        else:
            print(f"❌ /metrics Endpoint failed: {res.status_code}")
    except Exception as e:
        print(f"❌ /metrics Endpoint Exception: {e}")

    # 2. Predict Endpoint (Simulate)
    try:
        payload = {
            "pol": "USLAX",
            "pod": "CNSHA",
            "mode": "Ocean",
            "train": "false"
        }
        res = requests.post(f"{API_URL}/predict", data=payload)
        if res.status_code == 200:
            data = res.json()
            if 'predictions' in data and len(data['predictions']) > 0:
                print("✅ /predict Endpoint working (Simulation).")
                print(f"   Prediction: {data['predictions'][0]['PETA_Days']} days")
            else:
                 print("❌ /predict Endpoint returned unexpected structure.")
        else:
            print(f"❌ /predict Endpoint failed: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"❌ /predict Endpoint Exception: {e}")

if __name__ == "__main__":
    sf_ok = verify_snowflake()
    if sf_ok:
        verify_backend()
