import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG
import pandas as pd

conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
try:
    # Fetch 1 row to see all columns
    df = pd.read_sql("SELECT * FROM DT_INGESTION.FACT_TRIP LIMIT 1", conn)
    print("Columns found in FACT_TRIP:")
    for col in df.columns:
        print(f"- {col}")
finally:
    conn.close()
