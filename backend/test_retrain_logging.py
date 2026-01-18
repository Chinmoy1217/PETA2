from backend.main import retrain_model
import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG

def test_retrain_and_log():
    print("🚀 Triggering Model Retraining...")
    retrain_model()
    
    print("\n🧐 Verifying matching logs in Snowflake...")
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    try:
        cs = conn.cursor()
        cs.execute("USE SCHEMA DT_INGESTION")
        cs.execute("SELECT * FROM MODEL_ACCURACY_HISTORY ORDER BY TRAINING_TIME DESC LIMIT 1")
        row = cs.fetchone()
        if row:
            print("✅ Found recent accuracy log:")
            print(f"   - Time: {row[0]}")
            print(f"   - Model: {row[1]}")
            print(f"   - Accuracy: {row[2]}%")
            print(f"   - MAE: {row[3]}")
            print(f"   - R2: {row[4]}")
            print(f"   - Samples: {row[5]}")
        else:
            print("❌ No accuracy logs found in Snowflake.")
    except Exception as e:
        print(f"❌ Verification Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    test_retrain_and_log()
