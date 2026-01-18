import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG

def create_accuracy_table():
    print("❄️ Connecting to Snowflake...")
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cs = conn.cursor()
        
        # Use database and schema
        cs.execute(f"USE DATABASE {SNOWFLAKE_CONFIG['database']}")
        cs.execute("USE SCHEMA DT_INGESTION")
        
        print("🛠️ Creating MODEL_ACCURACY_HISTORY table...")
        create_sql = """
        CREATE TABLE IF NOT EXISTS MODEL_ACCURACY_HISTORY (
            TRAINING_TIME TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            MODEL_NAME STRING,
            ACCURACY_PCT FLOAT,
            MAE FLOAT,
            R2 FLOAT,
            SAMPLE_SIZE INTEGER
        )
        """
        cs.execute(create_sql)
        print("✅ Table created successfully.")
        
    except Exception as e:
        print(f"❌ Error creating table: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    create_accuracy_table()
