"""
Update PETA Results Table Schema - Add ATA vs PETA Tracking
"""
import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG

conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
cursor = conn.cursor()

# Drop old table
try:
    cursor.execute("DROP TABLE DT_INGESTION.PETA_PREDICTION_RESULTS")
    print("✅ Dropped old table")
except:
    print("ℹ️  No existing table to drop")

# Create new table with ATA vs PETA fields
create_sql = """
CREATE TABLE DT_INGESTION.PETA_PREDICTION_RESULTS (
    PREDICTION_ID VARCHAR(50) PRIMARY KEY,
    PREDICTION_TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    POL_CODE VARCHAR(10),
    POD_CODE VARCHAR(10),
    MODE_OF_TRANSPORT VARCHAR(20),
    CARRIER_SCAC_CODE VARCHAR(10),
    DEPARTURE_DATE DATE,
    CONGESTION_LEVEL NUMBER(3,0),
    WEATHER_SEVERITY NUMBER(3,0),
    PETA_HOURS NUMBER(10,2),
    PETA_DAYS NUMBER(10,2),
    PETA_DATE DATE,
    PETA_DATETIME TIMESTAMP_NTZ,
    ATA_DATE DATE,
    ATA_DATETIME TIMESTAMP_NTZ,
    ATA_HOURS NUMBER(10,2),
    ATA_DAYS NUMBER(10,2),
    VARIANCE_HOURS NUMBER(10,2),
    VARIANCE_DAYS NUMBER(10,2),
    ACCURACY_PCT NUMBER(5,2),
    MODEL_VERSION VARCHAR(50) DEFAULT 'Ensemble_v1',
    MODEL_ACCURACY_PCT NUMBER(5,2) DEFAULT 87.0,
    ROUTE VARCHAR(100),
    STATUS VARCHAR(20) DEFAULT 'PREDICTED',
    CREATED_BY VARCHAR(100) DEFAULT 'PETA_API',
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
"""

try:
    cursor.execute(create_sql)
    print("✅ Created table with ATA vs PETA fields")
    print()
    print("Schema includes:")
    print("  PETA_DATE/PETA_DATETIME - Predicted arrival")
    print("  ATA_DATE/ATA_DATETIME - Actual arrival") 
    print("  VARIANCE_HOURS/DAYS - ATA - PETA difference")
    print("  ACCURACY_PCT - Prediction accuracy")
    
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    cursor.close()
    conn.close()
