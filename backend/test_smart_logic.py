import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"
STATE_FILE = "model/data_state.json"

def test_1_initial_sync():
    print("\n[TEST 1] Initial Sync (State=0 rows)...")
    # Reset state
    with open(STATE_FILE, "w") as f:
        json.dump({"row_count": 0, "accuracy_pct": 0.0}, f)
        
    try:
        resp = requests.post(f"{BASE_URL}/sync-snowflake")
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
        
        if resp.json().get('status') == 'success' and 'Data Increased' in resp.json().get('message'):
            print("✅ Smart Trigger Worked (Retraining Started)")
        else:
            print("❌ Smart Trigger Failed")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_2_redundant_sync():
    print("\n[TEST 2] Redundant Sync (Should Skip)...")
    time.sleep(2) # Wait a bit for async tasks (mocking)
    
    # Simulate that previous retrain updated state to match 'new_count'
    # Since we can't wait for actual async retrain to finish in this script easily without polling,
    # We will manually set state to a high number to simulate 'up to date'.
    # Note: In real life, retrain updates this.
    
    # Get current count by force-check or assume a number. 
    # Let's peek at response 1 'new' count.
    # Hack: Set count to 999999 (likely higher than real) to force 'Has Changed' = False? 
    # No, 'Has Changed' is new > last. If last=999999, new(11) > 999999 is False.
    
    with open(STATE_FILE, "w") as f:
        json.dump({"row_count": 999999, "accuracy_pct": 87.0}, f)
        
    try:
        resp = requests.post(f"{BASE_URL}/sync-snowflake")
        print(f"Response: {resp.json()}")
        if resp.json().get('status') == 'skipped':
             print("✅ Smart Trigger Skipped (Data Unchanged)")
        else:
             print("❌ Failed to skip")
             
    except Exception as e: print(e)

def test_3_self_healing():
    print("\n[TEST 3] Self-Healing Guardrail...")
    # Set Baseline Accuracy to 99.9% (Impossible to beat)
    with open(STATE_FILE, "w") as f:
        json.dump({"row_count": 0, "accuracy_pct": 99.9}, f)
        
    # Force Retrain
    resp = requests.post(f"{BASE_URL}/sync-snowflake?force=true")
    print(f"Triggered Force Retrain: {resp.json()}")
    
    print("Wait for logs to show 'GUARDRAIL REJECTED'...")
    # In a real test harness we'd read logs. Here user will see it in terminal.

if __name__ == "__main__":
    test_1_initial_sync()
    test_2_redundant_sync()
    test_3_self_healing()
