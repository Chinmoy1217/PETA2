import sys
import os
import pandas as pd

# Mimic Notebook Path Setup
sys.path.append(os.getcwd())

print("--- Step 1: Imports ---")
try:
    from backend.data_loader import DataLoader
    from backend.model_engine import ETAModel
    from backend.quality_check import QualityCheck
    print("✅ Imports successful.")
except ImportError as e:
    print(f"❌ Import Failed: {e}")
    sys.exit(1)

print("\n--- Step 2: Initialize Loader & Connect ---")
try:
    loader = DataLoader()
    print("✅ Connected to Snowflake.")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    sys.exit(1)

print("\n--- Step 3: Fetch Data ---")
try:
    df_training = loader.get_training_view()
    print(f"✅ Loaded {len(df_training)} rows.")
    if 'Severity_Score' in df_training.columns:
        print("✅ Feature 'Severity_Score' found.")
    else:
        print("❌ 'Severity_Score' missing!")
except Exception as e:
    print(f"❌ Data Verification Failed: {e}")
    sys.exit(1)

print("\n--- Step 4: Quality Check ---")
try:
    dq = QualityCheck.run_checks(df_training)
    print(f"Quality Result: {dq['passed']}")
except Exception as e:
    print(f"❌ QA Failed: {e}")

print("\n--- Step 5: Train Model ---")
try:
    model_engine = ETAModel()
    res = model_engine.train()
    print(f"Training Result: {res['status']}")
except Exception as e:
    print(f"❌ Training Failed: {e}")

print("\n--- Step 6: Prediction ---")
try:
    # Test a prediction
    pred = model_engine.predict("CNDLC", "ARBUE", "OCEAN")
    print(f"Prediction: {pred['explanation']}")
except Exception as e:
    print(f"❌ Prediction Failed: {e}")
