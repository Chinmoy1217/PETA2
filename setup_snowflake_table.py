"""
Create PETA Results Table in Snowflake
"""
import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG

conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
cursor = conn.cursor()

# Create table
create_table_sql = """
CREATE TABLE IF NOT EXISTS DT_INGESTION.PETA_PREDICTION_RESULTS (
    PREDICTION_ID VARCHAR(50) PRIMARY KEY,
    PREDICTION_TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    POL_CODE VARCHAR(10),
    POD_CODE VARCHAR(10),
    MODE_OF_TRANSPORT VARCHAR(20),
    CARRIER_SCAC_CODE VARCHAR(10),
    DEPARTURE_DATE DATE,
    CONGESTION_LEVEL NUMBER(3,0),
    WEATHER_SEVERITY NUMBER(3,0),
    PREDICTED_HOURS NUMBER(10,2),
    PREDICTED_DAYS NUMBER(10,2),
    ARRIVAL_DATE DATE,
    ARRIVAL_DATETIME TIMESTAMP_NTZ,
    MODEL_VERSION VARCHAR(50) DEFAULT 'Ensemble_v1',
    MODEL_ACCURACY_PCT NUMBER(5,2) DEFAULT 87.0,
    BASE_ETA_DAYS NUMBER(10,2),
    ROUTE VARCHAR(100),
    CREATED_BY VARCHAR(100) DEFAULT 'PETA_API',
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
"""

try:
    cursor.execute(create_table_sql)
    print("✅ Table DT_INGESTION.PETA_PREDICTION_RESULTS created successfully")
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM DT_INGESTION.PETA_PREDICTION_RESULTS")
    count = cursor.fetchone()[0]
    print(f"📊 Current records in table: {count}")
    
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    cursor.close()
    conn.close()
