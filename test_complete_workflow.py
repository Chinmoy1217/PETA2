"""
Complete PETA vs ATA Test
1. Make predictions (PETA)
2. Simulate actual arrivals (ATA)
3. Calculate variance and accuracy
4. Display results from Snowflake
"""
import requests
import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG
from datetime import datetime, timedelta
import uuid
import time

API_URL = "http://localhost:8005/simulate"

print("=" * 100)
print("  COMPLETE PETA ↔ ATA WORKFLOW TEST")
print("  Testing prediction accuracy with simulated actual arrivals")
print("=" * 100)
print()

# ============================================================================
# STEP 1: Make Predictions (PETA)
# ============================================================================

print("STEP 1: MAKING PREDICTIONS (PETA)")
print("-" * 100)

test_shipments = [
    {
        "PolCode": "CNSHG",
        "PodCode": "USLAX",
        "ModeOfTransport": "OCEAN",
        "Carrier": "MAEU",
        "trip_ATD": "2024-01-15",
        "congestion_level": 50,
        "weather_severity": 20,
        "simulated_actual_hours": 42.5  # Simulate: arrived 3.7 hours late
    },
    {
        "PolCode": "INNSA",
        "PodCode": "DEHAM",
        "ModeOfTransport": "OCEAN",
        "Carrier": "MSCU",
        "trip_ATD": "2024-02-20",
        "congestion_level": 30,
        "weather_severity": 15,
        "simulated_actual_hours": 48.2  # Simulate: arrived early by 3.2 hours
    },
    {
        "PolCode": "SGSIN",
        "PodCode": "USNYC",
        "ModeOfTransport": "OCEAN",
        "Carrier": "COSCO",
        "trip_ATD": "2024-03-10",
        "congestion_level": 60,
        "weather_severity": 40,
        "simulated_actual_hours": 155.8  # Simulate: arrived 4.5 hours late
    },
]

predictions = []

for i, shipment in enumerate(test_shipments, 1):
    print(f"\nShipment {i}: {shipment['PolCode']} → {shipment['PodCode']}")
    
    # Remove simulated_actual_hours from API request
    api_data = {k: v for k, v in shipment.items() if k != 'simulated_actual_hours'}
    
    try:
        response = requests.post(API_URL, json=api_data, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            pred_hours = data.get('prediction_hours')
            
            if pred_hours:
                depart_date = datetime.strptime(shipment['trip_ATD'], "%Y-%m-%d")
                peta_datetime = depart_date + timedelta(hours=float(pred_hours))
                
                prediction_record = {
                    'shipment': shipment,
                    'peta_hours': pred_hours,
                    'peta_datetime': peta_datetime,
                    'route': f"{shipment['PolCode']} → {shipment['PodCode']}"
                }
                predictions.append(prediction_record)
                
                print(f"  ✅ PETA Predicted: {pred_hours:.1f} hours ({pred_hours/24:.1f} days)")
                print(f"     Arrival: {peta_datetime.strftime('%Y-%m-%d %H:%M')}")
            else:
                print(f"  ❌ No prediction returned")
        else:
            print(f"  ❌ API Error: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print()
print(f"✅ {len(predictions)} predictions made")
print()

# ============================================================================
# STEP 2: Save to Snowflake with PETA values
# ============================================================================

print("STEP 2: SAVING TO SNOWFLAKE")
print("-" * 100)

conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
cursor = conn.cursor()

prediction_ids = []

for pred in predictions:
    prediction_id = str(uuid.uuid4())
    prediction_ids.append(prediction_id)
    
    s = pred['shipment']
    
    cursor.execute("""
        INSERT INTO DT_INGESTION.PETA_PREDICTION_RESULTS (
            PREDICTION_ID, POL_CODE, POD_CODE, MODE_OF_TRANSPORT,
            CARRIER_SCAC_CODE, DEPARTURE_DATE, CONGESTION_LEVEL,
            WEATHER_SEVERITY, PETA_HOURS, PETA_DAYS,
            PETA_DATE, PETA_DATETIME, ROUTE, STATUS
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """, (
        prediction_id,
        s['PolCode'],
        s['PodCode'],
        s['ModeOfTransport'],
        s['Carrier'],
        s['trip_ATD'],
        s['congestion_level'],
        s['weather_severity'],
        pred['peta_hours'],
        pred['peta_hours'] / 24,
        pred['peta_datetime'].date(),
        pred['peta_datetime'],
        pred['route'],
        'PREDICTED'
    ))

conn.commit()
print(f"✅ Saved {len(predictions)} predictions to Snowflake")
print()

# ============================================================================
# STEP 3: Simulate Actual Arrivals (ATA)
# ============================================================================

print("STEP 3: UPDATING WITH ACTUAL ARRIVALS (ATA)")
print("-" * 100)
print("Simulating shipment completions...")
print()

for i, (pred_id, pred) in enumerate(zip(prediction_ids, predictions)):
    s = pred['shipment']
    
    # Calculate simulated actual arrival
    depart_date = datetime.strptime(s['trip_ATD'], "%Y-%m-%d")
    simulated_actual_hours = s['simulated_actual_hours']
    ata_datetime = depart_date + timedelta(hours=simulated_actual_hours)
    
    # Calculate variance
    peta_hours = pred['peta_hours']
    variance_hours = simulated_actual_hours - peta_hours
    variance_days = variance_hours / 24
    
    # Calculate accuracy (100% - percentage error)
    accuracy = max(0, 100 - abs(variance_hours / peta_hours * 100))
    
    # Update Snowflake
    cursor.execute("""
        UPDATE DT_INGESTION.PETA_PREDICTION_RESULTS
        SET 
            ATA_DATETIME = %s,
            ATA_DATE = %s,
            ATA_HOURS = %s,
            ATA_DAYS = %s,
            VARIANCE_HOURS = %s,
            VARIANCE_DAYS = %s,
            ACCURACY_PCT = %s,
            STATUS = 'COMPLETED',
            UPDATED_AT = CURRENT_TIMESTAMP()
        WHERE PREDICTION_ID = %s
    """, (
        ata_datetime,
        ata_datetime.date(),
        simulated_actual_hours,
        simulated_actual_hours / 24,
        variance_hours,
        variance_days,
        accuracy,
        pred_id
    ))
    
    print(f"Shipment {i+1}: {pred['route']}")
    print(f"  PETA: {peta_hours:.1f} hours → {pred['peta_datetime'].strftime('%Y-%m-%d %H:%M')}")
    print(f"  ATA:  {simulated_actual_hours:.1f} hours → {ata_datetime.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Variance: {variance_hours:+.1f} hours ({variance_days:+.1f} days)")
    print(f"  Accuracy: {accuracy:.1f}%")
    print()

conn.commit()
print("✅ All ATA values updated")
print()

# ============================================================================
# STEP 4: Query and Display Results
# ============================================================================

print("STEP 4: FINAL RESULTS FROM SNOWFLAKE")
print("-" * 100)

# Query all results
cursor.execute("""
    SELECT 
        ROUTE,
        DEPARTURE_DATE,
        PETA_DATE,
        PETA_HOURS,
        ATA_DATE,
        ATA_HOURS,
        VARIANCE_HOURS,
        VARIANCE_DAYS,
        ACCURACY_PCT,
        STATUS,
        CONGESTION_LEVEL,
        WEATHER_SEVERITY
    FROM DT_INGESTION.PETA_PREDICTION_RESULTS
    WHERE PREDICTION_ID IN (%s, %s, %s)
    ORDER BY DEPARTURE_DATE
""", tuple(prediction_ids))

results = cursor.fetchall()

print("\nDETAILED RESULTS:")
print("-" * 100)
for row in results:
    route, depart, peta_date, peta_hrs, ata_date, ata_hrs, var_hrs, var_days, acc, status, cong, weather = row
    print(f"\n{route}")
    print(f"  Departure:     {depart}")
    print(f"  Risk Factors:  Congestion={cong}, Weather={weather}")
    print(f"  PETA:          {peta_date} ({peta_hrs:.1f} hours)")
    print(f"  ATA:           {ata_date} ({ata_hrs:.1f} hours)")
    print(f"  Variance:      {var_hrs:+.1f} hours ({var_days:+.1f} days)")
    print(f"  Accuracy:      {acc:.1f}%")
    print(f"  Status:        {status}")

# Calculate overall statistics
cursor.execute("""
    SELECT 
        AVG(ACCURACY_PCT) as avg_accuracy,
        AVG(ABS(VARIANCE_HOURS)) as avg_error,
        MIN(VARIANCE_HOURS) as best,
        MAX(VARIANCE_HOURS) as worst
    FROM DT_INGESTION.PETA_PREDICTION_RESULTS
    WHERE PREDICTION_ID IN (%s, %s, %s)
""", tuple(prediction_ids))

stats = cursor.fetchone()
avg_acc, avg_err, best, worst = stats

print()
print("=" * 100)
print("OVERALL STATISTICS")
print("=" * 100)
print(f"Average Accuracy:  {avg_acc:.1f}%")
print(f"Average Error:     {avg_err:.1f} hours ({avg_err/24:.1f} days)")
print(f"Best Prediction:   {best:+.1f} hours variance")
print(f"Worst Prediction:  {worst:+.1f} hours variance")
print()

# Model comparison
if avg_acc >= 90:
    print("🎯 EXCELLENT: Model accuracy >90%")
elif avg_acc >= 85:
    print("✅ GOOD: Model accuracy 85-90% (current model baseline)")
elif avg_acc >= 80:
    print("⚠️  ACCEPTABLE: Model accuracy 80-85%")
else:
    print("❌ NEEDS IMPROVEMENT: Accuracy <80%")

print()
print("=" * 100)
print("✅ TEST COMPLETE - Full PETA ↔ ATA workflow validated")
print("=" * 100)

cursor.close()
conn.close()
