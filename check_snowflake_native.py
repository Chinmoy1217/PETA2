import snowflake.connector
import os
from backend.config import SNOWFLAKE_CONFIG

def test_conn():
    print("Testing Native Snowflake Connection...")
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT CURRENT_VERSION()")
        ver = cur.fetchone()[0]
        print(f"Connected! Snowflake Version: {ver}")
        
        cur.execute(f"USE DATABASE {SNOWFLAKE_CONFIG['database']}")
        cur.execute(f"USE SCHEMA {SNOWFLAKE_CONFIG['schema']}")
        
        # Verify FACT_TRIP exists (Main Data)
        cur.execute("SELECT COUNT(*) FROM DT_INGESTION.FACT_TRIP")
        count = cur.fetchone()[0]
        print(f"FACT_TRIP Row Count: {count}")
        conn.close()
    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    test_conn()
