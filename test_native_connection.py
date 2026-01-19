
import snowflake.connector
import sys
import os

# Add cwd to path so we can import backend
sys.path.append(os.getcwd())

from backend.config import SNOWFLAKE_CONFIG

print("Attempting Native Connection via Central Config...")

try:
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    
    print("✅ Connected!")
    cur = conn.cursor()
    cur.execute("SELECT CURRENT_VERSION()")
    print(f"Version: {cur.fetchone()[0]}")
    conn.close()
    
except Exception as e:
    print(f"❌ Connection Failed: {e}")
