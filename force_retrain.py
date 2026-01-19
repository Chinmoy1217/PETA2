
import os
import sys

# Add current dir to path to import backend
sys.path.append(os.getcwd())

from backend.main import retrain_model

print("Forcing Model Retraining to update Feature Schema...")
try:
    import pandas as pd
    if os.path.exists("local_full_data.pkl"):
        print("Loading local full dataset...")
        df = pd.read_pickle("local_full_data.pkl")
        retrain_model(external_df=df)
    else:
        print("Warning: local_full_data.pkl not found. Using default logic.")
        retrain_model()
        
    print("Retraining Complete. Model artifact updated.")
except Exception as e:
    print(f"Retraining Failed: {e}")
    import traceback
    traceback.print_exc()
