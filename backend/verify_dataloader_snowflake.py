from backend.data_loader import DataLoader
import pandas as pd

def verify_loader():
    print("🧪 Testing DataLoader with Snowflake...")
    loader = DataLoader()
    
    print("\n1. Testing get_active_trips()...")
    active_trips = loader.get_active_trips(limit=5)
    print(f"Found {len(active_trips)} active trips.")
    for trip in active_trips:
        print(f"   - {trip['id']}: {trip['origin']} -> {trip['destination']} ({trip['eta']})")
        
    print("\n2. Testing get_training_view()...")
    df = loader.get_training_view()
    print(f"Loaded training view with {len(df)} rows.")
    if not df.empty:
        print(f"Columns: {list(df.columns)}")
        print("First row sample:")
        print(df.iloc[0].to_dict())
    else:
        print("⚠️ Warning: Training view is empty. Check if FACT_TRIP and FACT_LANE_ETA_FEATURES have data.")

if __name__ == "__main__":
    verify_loader()
