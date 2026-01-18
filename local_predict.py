
import pandas as pd
import xgboost as xgb
import json
import os
import numpy as np

# Config
MODEL_DIR = "model"
DATA_FILE = "prediction_input.csv"

def predict_local():
    print(f"Loading Model from {MODEL_DIR}...")
    try:
        # 1. Load Model
        bst = xgb.Booster()
        bst.load_model(os.path.join(MODEL_DIR, "eta_xgboost.json"))
        
        # 2. Load Encoders
        with open(os.path.join(MODEL_DIR, "encoders.json"), "r") as f:
            encoders = json.load(f)

        # 3. Read Data
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded {len(df)} rows from {DATA_FILE}")
        
        # 4. Preprocess (Replicate main.py logic)
        col_map = {'POL': 'PolCode', 'POD': 'PodCode', 'Mode': 'ModeOfTransport'}
        df.rename(columns=col_map, inplace=True)
        
        # Add Missing Columns
        if 'External_Risk_Score' not in df.columns:
            df['External_Risk_Score'] = 0.0
            
        # Feature Engineering (Prepare Input)
        FEATURES = ['PolCode', 'PodCode', 'ModeOfTransport', 'Route']
        CAT_FEATURES = ['PolCode', 'PodCode', 'ModeOfTransport', 'Route']
        
        df['Route'] = df['PolCode'].astype(str) + "_" + df['PodCode'].astype(str) + "_" + df['ModeOfTransport'].astype(str)
        
        encoded_df = df.copy()
        for col in CAT_FEATURES:
            mapping = encoders.get(col, {})
            encoded_df[col] = encoded_df[col].astype(str).map(lambda s: mapping.get(s, 0))
            
        # Numeric
        # Check if model expects External_Risk_Score (It seems current artifact does not)
        # encoded_df['External_Risk_Score'] = pd.to_numeric(encoded_df['External_Risk_Score'], errors='coerce').fillna(0)
        
        # X_enc = encoded_df[FEATURES + ['External_Risk_Score']] 
        X_enc = encoded_df[FEATURES] # Revert to 4 features matching saved model

        
        # 5. Predict
        dtest = xgb.DMatrix(X_enc)
        preds = np.expm1(bst.predict(dtest))
        
        df['PETA_Predicted_Duration'] = np.round(preds, 2)
        
        # 6. Calculate Dates (ETA)
        if 'ATD' in df.columns:
            df['ATD'] = pd.to_datetime(df['ATD'])
            # Convert duration hours to timedelta
            df['Predicted_ATA'] = df['ATD'] + pd.to_timedelta(df['PETA_Predicted_Duration'], unit='h')
            
            print("\n--- PETA BATCH PREDICTIONS (WITH DATES) ---")
            print(df[['trip_id', 'PolCode', 'PodCode', 'ModeOfTransport', 'ATD', 'Predicted_ATA']].head(50).to_string(index=False))
        else:
            print("\n--- PETA BATCH PREDICTIONS (NO DATES) ---")
            print(df[['trip_id', 'PolCode', 'PodCode', 'ModeOfTransport', 'PETA_Predicted_Duration']].head(50).to_string(index=False))

        


        # --- FALLBACK: SAVE TO CSV ---
        output_csv = "batch_results_processed.csv"
        df[['trip_id', 'PolCode', 'PodCode', 'ModeOfTransport', 'ATD', 'PETA_Predicted_Duration', 'Predicted_ATA']].to_csv(output_csv, index=False)
        print(f"saved to {output_csv}")

        # Check against Snowflake Status (Mock check since script failed)
        print(f"\n[System Info] Local Prediction Complete.")
        

        # --- FALLBACK: GENERATE SQL SCRIPT ---
        sql_file = "batch_upload_script.sql"
        table_name = f"BATCH_RESULTS_MANUAL_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        with open(sql_file, "w") as f:
            f.write(f"-- Auto-Generated Upload Script for PETA Batch Predictions\n")
            f.write(f"CREATE TABLE IF NOT EXISTS {table_name} (\n")
            f.write(f"    TRIP_ID VARCHAR,\n")
            f.write(f"    POL VARCHAR,\n")
            f.write(f"    POD VARCHAR,\n")
            f.write(f"    MODE VARCHAR,\n")
            f.write(f"    ATD TIMESTAMP_NTZ,\n")
            f.write(f"    PETA_PREDICTED_DURATION FLOAT,\n")
            f.write(f"    ESTIMATED_ATA TIMESTAMP_NTZ\n")
            f.write(f");\n\n")
            
            for _, row in df.iterrows():
                val_str = f"('{row.get('trip_id', 'T0')}', '{row['PolCode']}', '{row['PodCode']}', '{row['ModeOfTransport']}', '{row['ATD']}', {row['PETA_Predicted_Duration']}, '{row['Predicted_ATA']}')"
                f.write(f"INSERT INTO {table_name} VALUES {val_str};\n")
                
        print(f"✅ GENERATED SQL SCRIPT: {sql_file}")
        print(f"Since direct connection failed, copy content of {sql_file} to Snowflake Worksheet.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
         if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    predict_local()
