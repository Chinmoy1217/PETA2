
import subprocess
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER_PATH = os.path.join(BASE_DIR, "backend", "snowflake_driver.js")

def run_query(sql):
    try:
        env = os.environ.copy()
        result = subprocess.run(
            ["node", DRIVER_PATH],
            input=sql,
            capture_output=True, text=True, env=env, encoding='utf-8'
        )
        if result.returncode != 0:
            print(f"FAILED: {result.stderr}")
            return []
        
        try:
            return json.loads(result.stdout)
        except:
            print(f"Raw Output: {result.stdout}")
            return []
            
    except Exception as e:
        print(f"Error: {e}")
        return []

print("--- 1. Tables in DT_INGESTION ---")
tables = run_query("SHOW TABLES IN SCHEMA DT_INGESTION")
for t in tables:
    print(f"- {t.get('name')} (Rows: {t.get('rows')}, Created: {t.get('created_on')})")

print("\n--- 2. Validating Training Tables ---")
target_tables = ['FACT_TRIP', 'DIM_LANE', 'DIM_VEHICLE', 'FACT_EXT_CONDITIONS']

for tt in target_tables:
    # Check if exists in list
    exists = any(row['name'] == tt for row in tables)
    if exists:
        # Get actual count
        count_res = run_query(f"SELECT COUNT(*) as CNT FROM {tt}")
        cnt = count_res[0]['CNT'] if count_res else "Error"
        print(f"✅ {tt}: Found. Count = {cnt}")
    else:
        print(f"❌ {tt}: NOT FOUND in DT_INGESTION schema!")
