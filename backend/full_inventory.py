
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
            return []
    except Exception as e:
        print(f"Error: {e}")
        return []

print("--- SEARCHING ALL SCHEMAS IN HACAKATHON ---")
# SHOW TABLES across the whole DB
tables = run_query("SHOW TABLES IN DATABASE HACAKATHON")

# Group by Schema for clarity
inventory = {}
for t in tables:
    schema = t.get('schema_name')
    if schema not in inventory: inventory[schema] = []
    inventory[schema].append({
        "name": t.get('name'),
        "rows": t.get('rows'),
        "created": t.get('created_on')
    })

for schema, tbls in inventory.items():
    print(f"\n[SCHEMA: {schema}]")
    for t in tbls:
        print(f"  - {t['name']} ({t['rows']} rows)")

print("\n--- GAP ANALYSIS: REQUIRED FOR PETA MODEL ---")
required = ['FACT_TRIP', 'DIM_LANE', 'DIM_VEHICLE', 'FACT_EXT_CONDITIONS', 'DIM_CARRIER']
found_map = {r: [] for r in required}

for t in tables:
    tname = t.get('name')
    if tname in required:
        found_map[tname].append(t.get('schema_name'))

for req, schemas in found_map.items():
    if schemas:
        print(f"✅ {req}: Found in {schemas}")
    else:
        print(f"❌ {req}: MISSING across entire database!")
