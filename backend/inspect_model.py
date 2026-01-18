import xgboost as xgb
import os
import json

MODEL_DIR = "model"
model_path = os.path.join(MODEL_DIR, "eta_xgboost.json")

try:
    bst = xgb.Booster()
    bst.load_model(model_path)
    print("✅ Model Loaded")
    print(f"Features ({len(bst.feature_names)}):")
    print(json.dumps(bst.feature_names, indent=2))
except Exception as e:
    print(f"❌ Error: {e}")
