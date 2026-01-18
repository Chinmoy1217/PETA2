
import snowflake.connector
import sys
import os

# Add cwd to path
sys.path.append(os.getcwd())
from backend.config import SNOWFLAKE_CONFIG

try:
    print("Connecting to Snowflake...")
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    print("✅ Connected. Creating Table...")
    cur = conn.cursor()
    
    # Create Table
    sql = """
    CREATE TABLE IF NOT EXISTS DT_PREDICTIONS (
        TRIP_ID VARCHAR(255),
        POL VARCHAR(50),
        POD VARCHAR(50),
        MODE VARCHAR(20),
        PREDICTED_DURATION FLOAT,
        PREDICTED_ATA TIMESTAMP_NTZ,
        CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
    )
    """
    cur.execute(sql)
    print("✅ Table DT_PREDICTIONS ensured.")
    conn.close()
    
except Exception as e:
    print(f"❌ DDL Failed: {e}")
