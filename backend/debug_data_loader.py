from backend.data_loader import DataLoader
import pandas as pd

print("--- DEBUGGING DATA LOADER ---")
try:
    loader = DataLoader()
    
    print("\n1. Trips Columns:")
    if loader.trips is not None:
        print(loader.trips.columns.tolist())
        print(f"Shape: {loader.trips.shape}")
    else:
        print("Trips is None")

    print("\n2. Carriers Columns:")
    if loader.carriers is not None:
        print(loader.carriers.columns.tolist())
    else:
        print("Carriers is None")
        
    print("\n3. Attempting Join Manually...")
    try:
        df = loader.trips.merge(loader.carriers, left_on='CID', right_on='CID')
        print("Join Successful!")
    except Exception as e:
        print(f"Join Failed: {e}")
        # Check for case mismatch
        if 'cid' in loader.trips.columns: print("Found lowercase 'cid' in Trips")
        if 'cid' in loader.carriers.columns: print("Found lowercase 'cid' in Carriers")

except Exception as e:
    print(f"Loader Init Failed: {e}")
