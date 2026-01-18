"""
Complete PETA Prediction Demo
Upload CSV → Get Arrival Predictions
"""
import requests
import pandas as pd
from datetime import datetime, timedelta

# Server URL
API_URL = "http://localhost:8005/simulate"

# Sample shipments to predict
shipments = [
    {"PolCode": "CNSHG", "PodCode": "USLAX", "ModeOfTransport": "OCEAN", "Carrier": "MAEU", "trip_ATD": "2024-06-15", "congestion_level": 50, "weather_severity": 20},
    {"PolCode": "INNSA", "PodCode": "DEHAM", "ModeOfTransport": "OCEAN", "Carrier": "MSCU", "trip_ATD": "2024-07-20", "congestion_level": 30, "weather_severity": 15},
    {"PolCode": "SGSIN", "PodCode": "USNYC", "ModeOfTransport": "OCEAN", "Carrier": "COSCO", "trip_ATD": "2024-08-10", "congestion_level": 60, "weather_severity": 40},
    {"PolCode": "AEJEA", "PodCode": "NLRTM", "ModeOfTransport": "OCEAN", "Carrier": "HMMU", "trip_ATD": "2024-09-05", "congestion_level": 25, "weather_severity": 10},
]

print("=" * 70)
print("  PETA PREDICTION SYSTEM - BATCH DEMO")
print("  Model: Ensemble (87% Accuracy)")
print("  Training Data: 118,652 Shipments")
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
                    'Carrier': shipment['Carrier'],
                    'Departure': shipment['trip_ATD'],
                    'PETA_Hours': round(float(pred_hours), 1),
                    'PETA_Days': round(float(pred_hours) / 24, 1),
                    'Arrival_Date': arrival_date.strftime('%Y-%m-%d'),
                    'Arrival_DateTime': arrival_date.strftime('%Y-%m-%d %H:%M')
                }
                
                results.append(result)
                
                print(f"✅ Shipment {i}: {result['Route']}")
                print(f"   Departure:  {result['Departure']}")
                print(f"   PETA:       {result['PETA_Hours']} hrs ({result['PETA_Days']} days)")
                print(f"   Arrival:    {result['Arrival_DateTime']}")
                print()
            else:
                print(f"⚠️  Shipment {i}: Response missing prediction field")
                print(f"   Available keys: {list(data.keys())}")
                print()
        else:
            print(f"❌ Shipment {i}: HTTP {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            print()
            
    except Exception as e:
        print(f"❌ Shipment {i}: {type(e).__name__}: {str(e)}")
        print()

# Save results
if results:
    df = pd.DataFrame(results)
    df.to_csv("peta_results.csv", index=False)
    print("=" * 70)
    print(f"✅ COMPLETE! {len(results)}/{len(shipments)} shipments predicted")
    print(f"📊 Results saved to: peta_results.csv")
    print("=" * 70)
else:
    print("=" * 70)
    print("❌ No predictions generated. Check API endpoint and data format.")
    print("=" * 70)
