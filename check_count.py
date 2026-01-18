
import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG

def check_count():
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM DT_INGESTION.FACT_TRIP")
        count = cur.fetchone()[0]
        print(f"Total Rows in Snowflake: {count}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    check_count()
