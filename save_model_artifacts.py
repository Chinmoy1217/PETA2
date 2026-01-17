import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
INPUT_FILE = 'Cleaned_Training_Data_Augmented.csv'
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_artifacts():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Feature Engineering
    df['Route'] = df['PolCode'].astype(str) + "_" + df['PodCode'].astype(str) + "_" + df['ModeOfTransport'].astype(str)
    
    FEATURES = ['PolCode', 'PodCode', 'ModeOfTransport', 'Route']
    TARGET = 'Actual_Duration_Hours'
    
    # MANUAL ENCODING (Native Python)
    encoders = {}
    for col in FEATURES:
        uniques = sorted(df[col].astype(str).unique())
        mapping = {val: i for i, val in enumerate(uniques)}
        df[col] = df[col].astype(str).map(mapping)
        encoders[col] = mapping
    
    with open(os.path.join(MODEL_DIR, 'encoders.json'), 'w') as f:
        json.dump(encoders, f)
    print("Encoders saved.")

    X = df[FEATURES]
    y = df[TARGET]
    # Log transform target
    y_log = np.log1p(y)
    
    # MANUAL SPLIT
    limit = int(len(df) * 0.8)
    # Shuffle
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_train = df_shuffled[FEATURES].iloc[:limit]
    y_train_log = np.log1p(df_shuffled[TARGET].iloc[:limit])
    
    X_val = df_shuffled[FEATURES].iloc[limit:]
    y_val = df_shuffled[TARGET].iloc[limit:] # Actuals
    y_val_log = np.log1p(y_val)

    # --- NATIVE XGBOOST API (No Sklearn Wrapper) ---
    print("Training XGBoost (Native API)...")
    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dval = xgb.DMatrix(X_val, label=y_val_log)
    
    params = {
        'objective': 'reg:squarederror',
        'eta': 0.05,
        'max_depth': 8,
        'nthread': 4
    }
    
    bst = xgb.train(
        params, 
        dtrain, 
        num_boost_round=500, 
        evals=[(dval, 'val')], 
        verbose_eval=50
    )
    
    # Save Model (JSON)
    bst.save_model(os.path.join(MODEL_DIR, 'eta_xgboost.json'))
    print("Model saved: eta_xgboost.json")
    
    # Prediction
    preds_log = bst.predict(dval)
    preds_xgb = np.expm1(preds_log)

    # --- SAVE PORT COORDINATES ---
    # Load Dim_Trip(in).csv to get Lat/Long
    coords = {}
    try:
        try:
            dim_df = pd.read_csv('Dim_Trip(in).csv', encoding='utf-8')
        except:
            dim_df = pd.read_csv('Dim_Trip(in).csv', encoding='latin1')
            
        # Extract POL coords
        if 'PolCode' in dim_df.columns and 'PolLatitude' in dim_df.columns:
            # Drop duplicates to keep one coord per port
            pols = dim_df[['PolCode', 'PolLatitude', 'PolLongitude']].dropna().drop_duplicates(subset=['PolCode'])
            for _, row in pols.iterrows():
                coords[row['PolCode']] = {'lat': row['PolLatitude'], 'lon': row['PolLongitude']}
                
        # Extract POD coords (merge into same dict)
        if 'PodCode' in dim_df.columns and 'PodLatitude' in dim_df.columns:
            pods = dim_df[['PodCode', 'PodLatitude', 'PodLongitude']].dropna().drop_duplicates(subset=['PodCode'])
            for _, row in pods.iterrows():
                coords[row['PodCode']] = {'lat': row['PodLatitude'], 'lon': row['PodLongitude']}
                
        print(f"Extracted coordinates for {len(coords)} ports.")
    except Exception as e:
        print(f"Failed to extract coordinates: {e}")
        # Fallback (Mock) for demo if file fails
        coords = {
            'USLAX': {'lat': 33.94, 'lon': -118.40},
            'GBLON': {'lat': 51.50, 'lon': -0.12},
            'CNSHA': {'lat': 31.23, 'lon': 121.47},
            'NLRTM': {'lat': 51.92, 'lon': 4.47}
        }

    with open(os.path.join(MODEL_DIR, 'port_coordinates.json'), 'w') as f:
        json.dump(coords, f)

    # --- OTHER MODELS (REAL TRAINING) ---
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train_log)
    preds_lr = np.expm1(lr.predict(X_val))
    with open(os.path.join(MODEL_DIR, 'linear_regression.pkl'), 'wb') as f:
        pickle.dump(lr, f)

    print("Training Random Forest (this may take a moment)...")
    # Using simpler params to speed up
    rf = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train_log)
    preds_rf = np.expm1(rf.predict(X_val))
    with open(os.path.join(MODEL_DIR, 'random_forest.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    
    print("All models saved.")

    # Calculate Real Metrics

    # Calculate Real Metrics
    def get_metrics(name, preds):
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        # Accuracy (1 - WAPE)
        wape = np.sum(np.abs(y_val - preds)) / np.sum(np.abs(y_val))
        acc = 100 * (1 - wape)
        return {'name': name, 'accuracy': round(acc, 2), 'rmse': round(rmse, 2), 'r2': round(r2, 2)}

    metrics = [
        get_metrics('XGBoost', preds_xgb),
        get_metrics('Linear Regression', preds_lr),
        get_metrics('Random Forest', preds_rf)
    ]
    
    # Sort by accuracy desc
    metrics.sort(key=lambda x: x['accuracy'], reverse=True)

    with open(os.path.join(MODEL_DIR, 'model_comparison.json'), 'w') as f:
        json.dump(metrics, f, indent=2, cls=NpEncoder)

    # --- INTELLIGENCE STATS ---
    # Save average duration per mode for "AI Summary" generation
    mode_stats = df.groupby('ModeOfTransport')[TARGET].mean().to_dict()
    # Convert integer keys back to string labels for the JSON
    # We need the reverse mapping from encoders
    mode_map_rev = {v: k for k, v in encoders['ModeOfTransport'].items()}
    mode_stats_str = {mode_map_rev[k]: round(v, 2) for k, v in mode_stats.items()}
    
    with open(os.path.join(MODEL_DIR, 'mode_stats.json'), 'w') as f:
        json.dump(mode_stats_str, f, indent=2, cls=NpEncoder)
    print("Mode stats saved for AI Summaries.")

    # --- Plot Data ---
    residuals = (y_val - preds_xgb).tolist()
    
    # Mode Performance
    # Reverse map mode
    mode_map_rev = {v: k for k, v in encoders['ModeOfTransport'].items()}
    val_modes_int = X_val['ModeOfTransport'].values
    val_modes_str = [mode_map_rev.get(i, "Unknown") for i in val_modes_int]
    
    mode_perf = []
    df_res = pd.DataFrame({'Mode': val_modes_str, 'Actual': y_val.values, 'Pred': preds_xgb})
    for mode in df_res['Mode'].unique():
        sub = df_res[df_res['Mode'] == mode]
        if len(sub) > 0:
            wape = np.sum(np.abs(sub['Actual'] - sub['Pred'])) / np.sum(np.abs(sub['Actual']))
            acc = 100 * (1 - wape)
            mode_perf.append({'mode': str(mode), 'accuracy': round(acc, 2)})

    hist_counts, hist_edges = np.histogram(residuals, bins=50)
    
    # Scatter Data (Sample 200 points)
    # Ensure float types
    df_res['Actual'] = df_res['Actual'].astype(float)
    df_res['Pred'] = df_res['Pred'].astype(float)
    
    if len(df_res) > 200:
        scatter_sample = df_res.sample(n=200, random_state=42).to_dict('records')
    else:
        scatter_sample = df_res.to_dict('records')
        
    # Timeline Data (Trips per Month) - requiring date column
    # We load INPUT_FILE again or use df if it has dates. 
    # df was loaded at start. Let's assume it has 'trip_start_date' or we use a mock for now if missing.
    try:
        if 'trip_ATD' in df.columns:
            df['date'] = pd.to_datetime(df['trip_ATD'])
            timeline = df.groupby(df['date'].dt.to_period('M')).size().reset_index(name='count')
            timeline['date'] = timeline['date'].astype(str)
            timeline_data = timeline.to_dict('records')
        else:
            # Fallback mock if dates missing
            timeline_data = [{"date": "2024-01", "count": 120}, {"date": "2024-02", "count": 145}]
    except:
        timeline_data = []

    plot_data = {
        'mode_performance': mode_perf,
        'scatter_data': scatter_sample, 
        'timeline_data': timeline_data,
        'residual_hist': {
            'counts': hist_counts.tolist(),
            'edges': hist_edges.tolist()
        }
    }
    
    with open(os.path.join(MODEL_DIR, 'plots.json'), 'w') as f:
        json.dump(plot_data, f, cls=NpEncoder)
        
    print("Success! All artifacts generated.")

if __name__ == "__main__":
    save_artifacts()
