import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Config
FEATURE_STORE_FILE = 'feature_store.pkl'
DEFAULT_OUTPUT_FILE = 'Processed_Inference_Data.csv'

def load_feature_store():
    try:
        with open(FEATURE_STORE_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {FEATURE_STORE_FILE} not found. Train the pipeline first.")
        return None

def process_data(input_csv):
    """
    Ingests a raw CSV with columns:
    shipment_id, origin, destination, carrier, departure_time, carrier_eta, service_type
    
    Outputs:
    Full Feature Matrix ready for Model Prediction.
    """
    print(f"Loading Inference Data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading input: {e}")
        return

    store = load_feature_store()
    if not store: return

    print("Hydrating Features from Store...")
    
    # 1. Normalize Input
    col_map = {c.lower(): c for c in df.columns}
    # map standard names
    # Expecting: shipment_id, origin, destination, carrier, departure_time, carrier_eta
    
    # Ensure datetimes
    df['departure_time'] = pd.to_datetime(df['departure_time'], errors='coerce')
    df['carrier_eta'] = pd.to_datetime(df['carrier_eta'], errors='coerce')
    
    # Fill missing basic
    df['service_type'] = df['service_type'].fillna('UNKNOWN').str.upper()
    df['carrier'] = df['carrier'].fillna('UNKNOWN')
    
    # 2. Derive Keys
    df['lane'] = df['origin'] + "-" + df['destination']
    
    # NEW: Via Port
    if 'via_port' not in df.columns:
        df['via_port'] = 'DIRECT'
    else:
        df['via_port'] = df['via_port'].fillna('DIRECT')
    
    # 3. Hydrate Lane Features (Avg Delay, Volatility)
    # We use map/apply with the dictionary for speed
    def get_lane_stat(lane, stat):
        stats = store['lane_stats'].get(lane, None)
        if stats: return stats.get(stat, 0)
        return 0 # Default for new lane
    
    df['lane_avg_delay'] = df['lane'].apply(lambda x: get_lane_stat(x, 'lane_avg_delay'))
    df['lane_delay_volatility'] = df['lane'].apply(lambda x: get_lane_stat(x, 'lane_delay_volatility'))
    df['lane_volume'] = df['lane'].apply(lambda x: get_lane_stat(x, 'lane_volume'))
    df['lane_avg_transit_hours'] = df['lane'].apply(lambda x: get_lane_stat(x, 'lane_avg_transit_hours'))
    
    # 4. Hydrate Carrier Features
    def get_carrier_stat(carrier, stat):
        stats = store['carrier_stats'].get(carrier, None)
        if stats: return stats.get(stat, 0)
        return 0
        
    df['carrier_avg_delay'] = df['carrier'].map(store['carrier_bias']).fillna(0)
    df['carrier_bias_hours'] = df['carrier_avg_delay'] # Same definition
    df['carrier_ontime_ratio'] = df['carrier'].apply(lambda x: get_carrier_stat(x, 'carrier_ontime_ratio'))
    df['carrier_volume'] = df['carrier'].apply(lambda x: get_carrier_stat(x, 'carrier_volume'))
    
    # 5. Time Features
    df['day_of_week'] = df['departure_time'].dt.dayofweek
    df['month'] = df['departure_time'].dt.month
    df['is_peak_season'] = df['month'].isin([10, 11, 12]).astype(int)
    
    # 6. Risk Scores
    df['origin_port_delay_index'] = df['origin'].map(store['origin_stats']).fillna(0)
    df['destination_port_delay_index'] = df['destination'].map(store['dest_stats']).fillna(0)
    # NEW: Via Port Risk (Multi-Hop)
    def get_inference_route_risk(via_str):
        if not via_str or pd.isna(via_str) or str(via_str) == 'DIRECT':
            return 0
        ports = str(via_str).split('|')
        risks = [store['via_stats'].get(p, 0) for p in ports]
        return np.mean(risks) if risks else 0

    df['via_port_congestion_index'] = df['via_port'].apply(get_inference_route_risk)
    
    # Route Risk Calculation
    max_vol = 50.0 # Approximate cap if unknown
    if 'max_vol' in store: max_vol = store.get('max_vol', 50.0) 
    
    vol_cap = df['lane_delay_volatility'].max() + 0.1
    if vol_cap < 1: vol_cap = 10.0
    
    df['route_risk_score'] = (
        (df['lane_delay_volatility'] / vol_cap) * 40 + 
        (df['destination_port_delay_index'].clip(0, 100) / 100) * 40 +
        (df['via_port_congestion_index'].clip(0, 100) / 100) * 20
    ).fillna(0)
    
    # Weather (Live Simulation Mock)
    df['weather_flag'] = (np.random.rand(len(df)) < 0.05).astype(int)
    
    # 7. Confidence Score
    # Vol Score
    lane_vol_max = store.get('max_vol', 1000)
    vol_score = np.log1p(df['lane_volume']) / np.log1p(lane_vol_max)
    
    # Volatility Score (Inverse)
    vol_capped = df['lane_delay_volatility'].clip(0, 48)
    stability_score = 1 - (vol_capped / 48)
    
    carrier_score = df['carrier_ontime_ratio']
    
    df['prediction_confidence'] = (stability_score * 0.4 + vol_score * 0.3 + carrier_score * 0.3).clip(0, 1)
    
    # Save
    print(f"Processed {len(df)} rows. Saving to {DEFAULT_OUTPUT_FILE}...")
    df.to_csv(DEFAULT_OUTPUT_FILE, index=False)
    print("Inference Data Ready.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_data(sys.argv[1])
    else:
        print("Usage: python process_new_data.py <input_csv>")
        # Test mode
        print("Running Test Mode with dummy input...")
        # Create a dummy csv
        dummy = pd.DataFrame({
            'shipment_id': ['TEST001', 'TEST002'],
            'origin': ['CNSHA', 'USLAX'],
            'destination': ['NLRTM', 'CNSHA'],
            'carrier': ['MAERSK', 'COSCO'],
            'via_port': ['SGSIN', 'DIRECT'],
            'departure_time': ['2025-05-01', '2025-06-01'],
            'carrier_eta': ['2025-05-20', '2025-06-20'],
            'service_type': ['OCEAN', 'OCEAN']
        })
        dummy.to_csv('test_input.csv', index=False)
        process_data('test_input.csv')
