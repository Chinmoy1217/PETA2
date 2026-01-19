"""
Quick verification of test results
"""
import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG

conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
cursor = conn.cursor()

print("=" * 100)
print("  VERIFICATION: Latest PETA vs ATA Results")
print("=" * 100)
print()

# Get latest completed shipments
cursor.execute("""
    SELECT 
        ROUTE,
        DEPARTURE_DATE,
        PETA_HOURS,
        PETA_DATE,
        ATA_HOURS,
        ATA_DATE,
        VARIANCE_HOURS,
        VARIANCE_DAYS,
        ACCURACY_PCT,
        CONGESTION_LEVEL,
        WEATHER_SEVERITY,
        STATUS,
        CREATED_AT
    FROM DT_INGESTION.PETA_PREDICTION_RESULTS
    WHERE STATUS = 'COMPLETED'
    ORDER BY CREATED_AT DESC
    LIMIT 5
""")

print("LATEST COMPLETED SHIPMENTS:")
print("-" * 100)

for row in cursor.fetchall():
    route, depart, peta_h, peta_d, ata_h, ata_d, var_h, var_d, acc, cong, weather, status, created = row
    
    print(f"\n{route}")
    print(f"  Departure:  {depart}")
    print(f"  Risk:       Congestion={cong}, Weather={weather}")
    print(f"  PETA:       {peta_d} ({peta_h:.1f}h)")
    print(f"  ATA:        {ata_d} ({ata_h:.1f}h)")
    print(f"  Variance:   {var_h:+.1f}h ({var_d:+.2f}d)")
    print(f"  Accuracy:   {acc:.1f}%")
    
print()
print("-" * 100)

# Overall stats
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN STATUS = 'COMPLETED' THEN 1 END) as completed,
        AVG(CASE WHEN STATUS = 'COMPLETED' THEN ACCURACY_PCT END) as avg_accuracy,
        AVG(CASE WHEN STATUS = 'COMPLETED' THEN ABS(VARIANCE_HOURS) END) as avg_error
    FROM DT_INGESTION.PETA_PREDICTION_RESULTS
""")

total, completed, avg_acc, avg_err = cursor.fetchone()

print()
print("OVERALL STATISTICS:")
print("-" * 100)
print(f"Total Predictions:     {total}")
print(f"Completed Shipments:   {completed}")
print(f"Average Accuracy:      {avg_acc or 0:.1f}%")
print(f"Average Error:         {avg_err or 0:.1f} hours ({(avg_err or 0)/24:.2f} days)")
print()

cursor.close()
conn.close()
