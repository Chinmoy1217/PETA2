
import requests
import json
import pandas as pd
from datetime import datetime, timedelta

# API Endpoint
url = "http://localhost:8005/simulate"

# Sample Data (CSV format)
# Columns: PolCode, PodCode, ModeOfTransport, Carrier, trip_ATD
data = [
    ["CNSHG", "USLAX", "OCEAN", "MAEU", "2024-02-01"], # Shanghai -> LA (Maersk) in Feb
    ["INNSA", "DEHAM", "OCEAN", "MSCU", "2024-03-15"], # Nhava Sheva -> Hamburg (MSC) in Mar
    ["CNYTN", "NSNYC", "OCEAN", "CMAU", "2024-05-20"]  # Yantian -> NYC (CMA CGM) in May
]

# Create CSV
df = pd.DataFrame(data, columns=["PolCode", "PodCode", "ModeOfTransport", "Carrier", "trip_ATD"])
df.to_csv("test_samples.csv", index=False)
print("Created 'test_samples.csv'")

# Loop and Predict
print("\n--- PETA PREDICTIONS ---")
for index, row in df.iterrows():
    # Construct Payload
    payload = {
        "PolCode": row['PolCode'],
        "PodCode": row['PodCode'],
        "ModeOfTransport": row['ModeOfTransport'],
        "Carrier": row['Carrier'],
        "trip_ATD": row['trip_ATD']
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            res = response.json()
            peta_hours = res['predicted_duration_hours']
            
            # Calculate Dates
            start_date = datetime.strptime(row['trip_ATD'], "%Y-%m-%d")
            arrival_date = start_date + timedelta(hours=peta_hours)
            
            print(f"\nTrip: {row['PolCode']} -> {row['PodCode']} ({row['Carrier']})")
            print(f"  Departure: {row['trip_ATD']}")
            print(f"  PETA (Hours): {peta_hours:.2f} hrs")
            print(f"  PETA (Date):  {arrival_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Base ETA Days: {res.get('base_eta_days', 'N/A')}")
            print(f"  Risk Factors: {res.get('risk_factors', {})}")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Connection Failed: {e}")
