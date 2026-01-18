"""
Verify PETA Results in Snowflake
"""
import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG

conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
cursor = conn.cursor()

try:
    # Count records
    cursor.execute("SELECT COUNT(*) FROM DT_INGESTION.PETA_PREDICTION_RESULTS")
    count = cursor.fetchone()[0]
    print(f"📊 Total predictions in Snowflake: {count}")
    print()
    
    # Show latest predictions
    cursor.execute("""
        SELECT 
            POL_CODE,
            POD_CODE,
            CARRIER_SCAC_CODE,
            DEPARTURE_DATE,
            PREDICTED_HOURS,
            PREDICTED_DAYS,
            ARRIVAL_DATE,
            CONGESTION_LEVEL,
            WEATHER_SEVERITY,
            PREDICTION_TIMESTAMP
        FROM DT_INGESTION.PETA_PREDICTION_RESULTS
        ORDER BY PREDICTION_TIMESTAMP DESC
        LIMIT 10
    """)
    
    results = cursor.fetchall()
    
    if results:
        print("Latest Predictions:")
        print("-" * 100)
        for row in results:
            print(f"{row[0]} → {row[1]} | {row[2]} | Depart: {row[3]} | PETA: {row[4]}hrs ({row[5]}days) | Arrive: {row[6]} | Risk: C={row[7]} W={row[8]}")
    else:
        print("No predictions found")
    
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    cursor.close()
    conn.close()
