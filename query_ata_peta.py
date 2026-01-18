"""
Query Examples for ATA vs PETA Analysis
"""
import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG

conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
cursor = conn.cursor()

print("=" * 80)
print("  ATA vs PETA ANALYSIS - SNOWFLAKE QUERIES")
print("=" * 80)
print()

# 1. Show all predictions with their status
print("1. ALL PREDICTIONS (PETA vs ATA)")
print("-" * 80)
cursor.execute("""
    SELECT 
        POL_CODE || ' → ' || POD_CODE AS ROUTE,
        CARRIER_SCAC_CODE,
        DEPARTURE_DATE,
        PETA_DATE,
        PETA_HOURS,
        ATA_DATE,
        ATA_HOURS,
        VARIANCE_HOURS,
        ACCURACY_PCT,
        STATUS
    FROM DT_INGESTION.PETA_PREDICTION_RESULTS
    ORDER BY CREATED_AT DESC
    LIMIT 10
""")

for row in cursor.fetchall():
    route, carrier, depart, peta_date, peta_hrs, ata_date, ata_hrs, var_hrs, acc, status = row
    print(f"{route:20} | {carrier:5} | Depart: {depart} | PETA: {peta_date} ({peta_hrs:.1f}h) | " +
          f"ATA: {ata_date or 'Pending'} | Variance: {var_hrs or 'N/A'} | Status: {status}")

print()
print("=" * 80)
print()

# 2. Calculate overall accuracy
print("2. OVERALL MODEL ACCURACY")
print("-" * 80)
cursor.execute("""
    SELECT 
        COUNT(*) AS TOTAL_PREDICTIONS,
        COUNT(ATA_DATE) AS COMPLETED_SHIPMENTS,
        AVG(ACCURACY_PCT) AS AVG_ACCURACY,
        AVG(ABS(VARIANCE_HOURS)) AS AVG_ERROR_HOURS,
        MIN(VARIANCE_HOURS) AS BEST_PREDICTION,
        MAX(VARIANCE_HOURS) AS WORST_PREDICTION
    FROM DT_INGESTION.PETA_PREDICTION_RESULTS
""")

result = cursor.fetchone()
print(f"Total Predictions:    {result[0]}")
print(f"Completed Shipments:  {result[1]}")
print(f"Average Accuracy:     {result[2] or 0:.1f}%")
print(f"Average Error:        {result[3] or 0:.1f} hours")
print(f"Best Prediction:      {result[4] or 0:+.1f} hours variance")
print(f"Worst Prediction:     {result[5] or 0:+.1f} hours variance")

print()
print("=" * 80)
print()

# 3. Route-level accuracy
print("3. ACCURACY BY ROUTE")
print("-" * 80)
cursor.execute("""
    SELECT 
        ROUTE,
        COUNT(*) AS PREDICTIONS,
        AVG(PETA_HOURS) AS AVG_PETA,
        AVG(ATA_HOURS) AS AVG_ATA,
        AVG(VARIANCE_HOURS) AS AVG_VARIANCE,
        AVG(ACCURACY_PCT) AS AVG_ACCURACY
    FROM DT_INGESTION.PETA_PREDICTION_RESULTS
    WHERE ATA_DATE IS NOT NULL
    GROUP BY ROUTE
    ORDER BY AVG_ACCURACY DESC
""")

for row in cursor.fetchall():
    route, count, avg_peta, avg_ata, avg_var, avg_acc = row
    if avg_ata:
        print(f"{route:25} | {count:3} trips | PETA: {avg_peta:.1f}h | ATA: {avg_ata:.1f}h | " +
              f"Variance: {avg_var:+.1f}h | Accuracy: {avg_acc:.1f}%")

print()
print("=" * 80)

cursor.close()
conn.close()
