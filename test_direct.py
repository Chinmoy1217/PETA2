
import xgboost as xgb
import os
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join("model")
MODEL_PATH = os.path.join(MODEL_DIR, "eta_xgboost.json")

print(f"Loading model from {MODEL_PATH}...")
bst = xgb.Booster()
bst.load_model(MODEL_PATH)

print(f"Model Feature Names: {bst.feature_names}")

# Dummy Input using Global Features List
FEATURES = ['PolCode', 'PodCode', 'ModeOfTransport', 'Route', 'Carrier',
            'External_Risk_Score', 'BASE_ETA_DAYS', 'WEATHER_RISK_SCORE',
            'GEOPOLITICAL_RISK_SCORE', 'LABOR_STRIKE_SCORE', 'CUSTOMS_DELAY_SCORE', 
            'PORT_CONGESTION_SCORE', 'CARRIER_DELAY_SCORE', 'PEAK_SEASON_SCORE',
            'Risk_Intensity', 'Peak_Risk_Factor', 'Route_Target_Enc', 'Carrier_Target_Enc',
            'Departure_Month', 'Departure_Week'] 

print(f"\nExpected Features ({len(FEATURES)}): {FEATURES}")

# Create random data
df = pd.DataFrame(np.random.rand(1, len(FEATURES)), columns=FEATURES)

# Create DMatrix
dmat = xgb.DMatrix(df)

print("\nPredicting...")
try:
    pred = bst.predict(dmat)
    print(f"Prediction Success: {pred[0]}")
except Exception as e:
    print(f"Prediction Failed: {e}")
