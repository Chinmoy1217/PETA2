
import subprocess
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER_PATH = os.path.join(BASE_DIR, "backend", "snowflake_driver.js")

def test():
    query = "SELECT CURRENT_VERSION()"
    print(f"Testing Driver with Query: {query}")
    print(f"Driver Path: {DRIVER_PATH}")
    
    try:
        result = subprocess.run(
            ["node", DRIVER_PATH],
            input=query,
            capture_output=True, text=True, encoding='utf-8'
        )
        
        if result.returncode != 0:
            print("FAILED. Stderr:")
            print(result.stderr)
        else:
            print("SUCCESS. Stdout:")
            print(result.stdout)
            try:
                data = json.loads(result.stdout)
                print(f"Parsed JSON: {data}")
            except:
                print("Could not parse JSON.")
                
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test()
