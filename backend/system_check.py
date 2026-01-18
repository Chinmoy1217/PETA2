
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def log(msg): 
    print(f"[TEST] {msg}")

def check_health():
    print("\n--- 1. API HEALTH CHECK ---")
    try:
        resp = requests.get(f"{BASE_URL}/")
        if resp.status_code == 200:
            log(f"✅ API is Online & Reachable: {resp.json()}")
            return True
        else:
            log(f"❌ API Error: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        log(f"❌ Connection Failed: {e}")
        return False

def check_inference():
    print("\n--- 2. MODEL INFERENCE CHECK ---")
    # Using 'data' sends form-encoded (compatible with File+Body mix usually)
    # or separate args.
    # Given main.py uses Body(None), let's try JSON first, but if 400, try Form.
    # Actually, main.py with File=File(None) usually implies Multipart if mixed.
    # But let's try sending as standard JSON body first, using requests' json param.
    payload = {
        "pol": "CNSHG", 
        "pod": "USLAX", 
        "mode": "OCEAN"
    }
    
    try:
        # Try JSON
        resp = requests.post(f"{BASE_URL}/predict", json=payload)
        
        if resp.status_code == 200:
            log(f"✅ Prediction Success: {resp.json().get('peta_eta_hours')} hours")
            log(f"   -> Explanation: {resp.json().get('ai_explanation')}")
        elif resp.status_code == 422: # Validation Error
             log(f"⚠️ Validation Error: {resp.text}")
        elif resp.status_code == 400: # Our Custom Logic Error
             # Fallback: Try Form Data (if endpoint expects multipart because of File)
             log("ℹ️ JSON Rejected. Retrying with Form Data/Query...")
             resp_form = requests.post(f"{BASE_URL}/predict", data=payload) # Form Data
             if resp_form.status_code == 200:
                  log(f"✅ Prediction Success (Form): {resp_form.json().get('peta_eta_hours')} hours")
             else:
                  log(f"❌ Inference Failed (Form): {resp_form.status_code} - {resp_form.text}")
        else:
             log(f"❌ Inference Failed: {resp.status_code} - {resp.text}")
             
    except Exception as e:
        log(f"❌ Request Error: {e}")

def check_snowflake_training():
    print("\n--- 3. SNOWFLAKE TRAINING PIPELINE CHECK ---")
    log("Calling /sync-snowflake to pull data from DB...")
    
    try:
        resp = requests.post(f"{BASE_URL}/sync-snowflake")
        if resp.status_code == 200:
            data = resp.json()
            log(f"✅ Response: {data}")
            if "status" in data and data["status"] == "success":
                 log("   -> PROOF: Successfully talked to Snowflake.")
            else:
                 log("   ⚠️  Warning: Check message details.")
        else:
            log(f"❌ Sync Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        log(f"❌ Sync Error: {e}")

if __name__ == "__main__":
    if check_health():
        check_inference()
        check_snowflake_training()
