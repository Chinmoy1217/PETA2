
import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG

def inspect():
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cur = conn.cursor()
        cur.execute("DESCRIBE TABLE DT_INGESTION.FACT_TRIP")
        
        print(f"{'Column':<30} {'Type':<20}")
        print("-" * 50)
        for row in cur.fetchall():
            print(f"{row[0]:<30} {row[1]:<20}")
            
    except Exception as e:
        print(e)
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    inspect()
