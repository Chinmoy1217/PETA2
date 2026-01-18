"""
PETA Predictions with Snowflake Integration
Predicts ETAs and saves results to Snowflake
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG
import uuid

# API Configuration
API_URL = "http://localhost:8005/simulate"

# Snowflake Connection
def get_snowflake_connection():
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Snowflake Connection Failed: {e}")
        return None

# Insert results into Snowflake
def save_to_snowflake(results_df):
    """Save prediction results to Snowflake table"""
    conn = get_snowflake_connection()
    if not conn:
        print("⚠️  Cannot save to Snowflake - connection failed")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Prepare insert statement
        insert_sql = """
        INSERT INTO DT_INGESTION.PETA_PREDICTION_RESULTS (
            PREDICTION_ID, POL_CODE, POD_CODE, MODE_OF_TRANSPORT, 
            CARRIER_SCAC_CODE, DEPARTURE_DATE, CONGESTION_LEVEL, 
            WEATHER_SEVERITY, PETA_HOURS, PETA_DAYS, 
            PETA_DATE, PETA_DATETIME, ROUTE, STATUS
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        # Insert each row
        for _, row in results_df.iterrows():
            prediction_id = str(uuid.uuid4())
            
            # Parse route
            pol, pod = row['Route'].split(' → ')
            
            cursor.execute(insert_sql, (
                prediction_id,
                pol,
                pod,
                row.get('Mode', 'OCEAN'),
                row['Carrier'],
                row['Departure'],
                row.get('Congestion', 0),
                row.get('Weather', 0),
                row['PETA_Hours'],
                row['PETA_Days'],
                row['Arrival_Date'],
                row['Arrival_DateTime'],
                row['Route'],
                'PREDICTED'  # STATUS
            ))
        
        conn.commit()
        print(f"✅ Saved {len(results_df)} predictions to Snowflake table: DT_INGESTION.PETA_PREDICTION_RESULTS")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error saving to Snowflake: {e}")
        import traceback
        traceback.print_exc()
        return False

# Sample shipments to predict
shipments = [
    {"PolCode": "CNSHG", "PodCode": "USLAX", "ModeOfTransport": "OCEAN", "Carrier": "MAEU", "trip_ATD": "2024-06-15", "congestion_level": 50, "weather_severity": 20},
    {"PolCode": "INNSA", "PodCode": "DEHAM", "ModeOfTransport": "OCEAN", "Carrier": "MSCU", "trip_ATD": "2024-07-20", "congestion_level": 30, "weather_severity": 15},
    {"PolCode": "SGSIN", "PodCode": "USNYC", "ModeOfTransport": "OCEAN", "Carrier": "COSCO", "trip_ATD": "2024-08-10", "congestion_level": 60, "weather_severity": 40},
    {"PolCode": "AEJEA", "PodCode": "NLRTM", "ModeOfTransport": "OCEAN", "Carrier": "HMMU", "trip_ATD": "2024-09-05", "congestion_level": 25, "weather_severity": 10},
]

print("=" * 70)
print("  PETA PREDICTION SYSTEM - SNOWFLAKE INTEGRATION")
print("  Model: Ensemble (87% Accuracy)")
print("  Destination: DT_INGESTION.PETA_PREDICTION_RESULTS")
print("=" * 70)
print()

results = []

for i, shipment in enumerate(shipments, 1):
    try:
        response = requests.post(API_URL, json=shipment, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract prediction
            pred_hours = (
                data.get('prediction_hours') or
                data.get('predicted_duration_hours') or
                data.get('predicted_eta_hours') or  
                data.get('final_prediction') or
                data.get('prediction')
            )
            
            if pred_hours:
                # Calculate arrival date
                depart_date = datetime.strptime(shipment['trip_ATD'], "%Y-%m-%d")
                arrival_date = depart_date + timedelta(hours=float(pred_hours))
                
                result = {
                    'Route': f"{shipment['PolCode']} → {shipment['PodCode']}",
                    'Mode': shipment['ModeOfTransport'],
                    'Carrier': shipment['Carrier'],
                    'Departure': shipment['trip_ATD'],
                    'Congestion': shipment['congestion_level'],
                    'Weather': shipment['weather_severity'],
                    'PETA_Hours': round(float(pred_hours), 1),
                    'PETA_Days': round(float(pred_hours) / 24, 1),
                    'Arrival_Date': arrival_date.strftime('%Y-%m-%d'),
                    'Arrival_DateTime': arrival_date.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                results.append(result)
                
                print(f"✅ Shipment {i}: {result['Route']}")
                print(f"   Departure:  {result['Departure']}")
                print(f"   PETA:       {result['PETA_Hours']} hrs ({result['PETA_Days']} days)")
                print(f"   Arrival:    {result['Arrival_DateTime']}")
                print(f"   Conditions: Congestion={result['Congestion']}, Weather={result['Weather']}")
                print()
            else:
                print(f"⚠️  Shipment {i}: Response missing prediction field")
                print()
        else:
            print(f"❌ Shipment {i}: HTTP {response.status_code}")
            print()
            
    except Exception as e:
        print(f"❌ Shipment {i}: {type(e).__name__}: {str(e)}")
        print()

# Save results
if results:
    df = pd.DataFrame(results)
    
    # Save to CSV (local backup)
    df.to_csv("peta_results.csv", index=False)
    print("=" * 70)
    print(f"✅ {len(results)}/{len(shipments)} shipments predicted")
    print(f"📊 Local CSV saved: peta_results.csv")
    
    # Save to Snowflake
    print()
    print("Saving to Snowflake...")
    if save_to_snowflake(df):
        print("=" * 70)
        print("🎯 SUCCESS! Results saved to both CSV and Snowflake")
        print("=" * 70)
    else:
        print("=" * 70)
        print("⚠️  Predictions completed but Snowflake save failed")
        print("=" * 70)
else:
    print("=" * 70)
    print("❌ No predictions generated")
    print("=" * 70)
