import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG

def check_update():
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cs = conn.cursor()
        cs.execute(f"USE SCHEMA {SNOWFLAKE_CONFIG['schema']}")
        cs.execute("SELECT * FROM MODEL_ACCURACY_HISTORY ORDER BY TRAINING_TIME DESC LIMIT 1")
        row = cs.fetchone()
        if row:
            print(f"Latest Log: Model={row[1]}, Acc={row[2]}%, Time={row[0]}")
        else:
            print("No logs found.")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_update()
