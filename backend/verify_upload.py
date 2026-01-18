
import sys, types
from collections import abc
import collections

# --- MONKEY PATCHES for Python 3.13 ---
m = types.ModuleType("cgi")
m.parse_header = lambda x: (x, {})
sys.modules["cgi"] = m

collections.Mapping = abc.Mapping
collections.MutableMapping = abc.MutableMapping
collections.Sequence = abc.Sequence
# --------------------------------------

import snowflake.connector
import os

# Credentials (Recovered)
USER = "HACKATHON_DT"
PASSWORD = "Welcome@Start123"
ACCOUNT = "COZENTUS-DATAPRACTICE"
DATABASE = "HACAKATHON"
SCHEMA = "PUBLIC"

def verify():
    print("Connecting to Snowflake...")
    try:
        conn = snowflake.connector.connect(
            user=USER,
            password=PASSWORD,
            account=ACCOUNT,
            database=DATABASE,
            schema=SCHEMA
        )
        cur = conn.cursor()
        
        # Find latest BATCH_RESULTS table
        print("Searching for BATCH_RESULTS tables...")
        cur.execute("SHOW TABLES LIKE 'BATCH_RESULTS_%'")
        tables = cur.fetchall()
        
        if not tables:
            print("No BATCH_RESULTS tables found.")
            return
            
        # Sort by creation time (assuming name contains timestamp, but SHOW TABLES gives us 'created_on')
        # tables structure depends on Snowflake version, but robustly:
        print(f"Found {len(tables)} tables.")
        
        for t in tables:
            t_name = t[1] # Name is usually index 1
            print(f"Checking {t_name}...")
            
            # Check count
            cur.execute(f"SELECT COUNT(*) FROM {t_name}")
            count = cur.fetchone()[0]
            print(f"Table: {t_name}, Rows: {count}, Status: {'SUCCESS' if count >= 50 else 'PARTIAL'}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    verify()
