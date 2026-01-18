import requests

# Upload CSV and get PETA predictions
url = "http://localhost:8005/predict"

print("Testing Batch PETA Prediction via File Upload...")
print("-" * 60)

# Upload CSV file
with open("sample_shipments.csv", "rb") as f:
    files = {"file": ("sample_shipments.csv", f, "text/csv")}
    data = {"train": "false"}  # Don't trigger retraining
    
    try:
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS! Processed {len(result.get('predictions', []))} shipments\n")
            
            # Display results
            for i, pred in enumerate(result.get('predictions', []), 1):
                print(f"Shipment {i}:")
                print(f"  Route: {pred.get('origin')} → {pred.get('destination')}")
                print(f"  PETA: {pred.get('predicted_eta_hours', 0):.1f} hours ({pred.get('predicted_eta_hours', 0)/24:.1f} days)")
                print(f"  Base ETA: {pred.get('base_eta', 'N/A')}")
                print()
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
