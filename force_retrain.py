
import os
import sys

# Add current dir to path to import backend
sys.path.append(os.getcwd())

from backend.main import retrain_model

print("Forcing Model Retraining to update Feature Schema...")
try:
    # This will load data (csv + snowflake), add External_Risk_Score, and overwrite json model
    retrain_model()
    print("Retraining Complete. Model artifact updated.")
except Exception as e:
    print(f"Retraining Failed: {e}")
