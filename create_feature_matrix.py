import pandas as pd
import numpy as np

# Config
INPUT_FILE = 'Cleaned_Training_Data_Augmented.csv'
OUTPUT_FILE = 'ETA_FEATURES_FINAL.csv'

def generate_features():
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please run previous pipeline steps first.")
        return

    print("Generating Feature Matrix (Hackathon Blueprint)...")
    
    # --- 1. RAW INPUT COLUMNS ---
    # shipment_id, origin, destination, carrier, departure_time, carrier_eta, actual_arrival
    
    df['shipment_id'] = df['trip_id'].fillna(df.index.to_series().astype(str))
    df['origin'] = df['PolCode']
    df['destination'] = df['PodCode']
    
    # NEW: Via Port (Transshipment)
    if 'via_port' in df.columns:
        df['via_port'] = df['via_port'].fillna('DIRECT')
    else:
        df['via_port'] = 'DIRECT'
        
    df['departure_time'] = pd.to_datetime(df['trip_ATD'], errors='coerce')
    df['actual_arrival'] = pd.to_datetime(df['trip_ATA'], errors='coerce')
    
    # Ensure Service Type matches carrier logic roughly (for demo visual)
    df['service_type'] = df['ModeOfTransport'].str.upper().str.strip()
    
    # Carrier Extraction
    if 'Transport_Vehicle_ID' in df.columns:
        df['carrier'] = df['Transport_Vehicle_ID'].apply(lambda x: str(x).split(' ')[0] if pd.notna(x) else 'UNKNOWN')
    else:
        carriers = ['MAERSK', 'MSC', 'CMA', 'HAPAG', 'ONE', 'FEDEX', 'DHL', 'UPS']
        df['carrier'] = np.random.choice(carriers, size=len(df))

    # Carrier ETA Simulation (The "Promise")
    # In reality, carrier_eta is given. Here we simulate it as: Actual - Random Delay Deviation
    
    # Delay Distribution
    noise = np.random.gamma(shape=1.5, scale=12.0, size=len(df)) 
    
    # Mask some as 0 (On Time) - 60%
    mask_ontime = np.random.rand(len(df)) < 0.6
    noise[mask_ontime] = np.random.uniform(-0.5, 0.5, size=mask_ontime.sum())
    
    # Add some negatives (Early arrival) - 10%
    mask_early = np.random.rand(len(df)) < 0.1
    noise[mask_early] = -np.random.uniform(1, 24, size=mask_early.sum())
    
    # Assign delay
    df['delay_hours'] = noise
    
    # carrier_eta = actual - delay
    df['carrier_eta'] = df['actual_arrival'] - pd.to_timedelta(df['delay_hours'], unit='h')
    
    # Correction: ensure carrier_eta is > departure_time
    # If delay makes eta before departure (impossible), fix it.
    mask_impossible = df['carrier_eta'] <= df['departure_time']
    df.loc[mask_impossible, 'carrier_eta'] = df.loc[mask_impossible, 'departure_time'] + pd.Timedelta(hours=24)
    # Recalculate delay for these fixed rows
    df.loc[mask_impossible, 'delay_hours'] = (df.loc[mask_impossible, 'actual_arrival'] - df.loc[mask_impossible, 'carrier_eta']).dt.total_seconds() / 3600

    # --- 2. TARGET COLUMN ---
    # delay_hours (Already done above to ensure consistency)
    
    # --- 3. ENGINEERED / COOKED FEATURES ---
    
    # A. Lane Features
    df['lane'] = df['origin'] + "-" + df['destination']
    
    print("Calculating Lane Metrics...")
    lane_stats = df.groupby('lane')['delay_hours'].agg(['mean', 'std', 'count']).reset_index()
    lane_stats.columns = ['lane', 'lane_avg_delay', 'lane_delay_volatility', 'lane_volume']
    
    # Fill volatility nan with 0 (single trip lanes)
    lane_stats['lane_delay_volatility'] = lane_stats['lane_delay_volatility'].fillna(0)
    
    # Lane Avg Transit
    df['transit_hours'] = (df['actual_arrival'] - df['departure_time']).dt.total_seconds() / 3600
    lane_transit = df.groupby('lane')['transit_hours'].mean().reset_index(name='lane_avg_transit_hours')
    
    df = df.merge(lane_stats, on='lane', how='left')
    df = df.merge(lane_transit, on='lane', how='left')
    
    # B. Carrier Performance Features
    print("Calculating Carrier Metrics...")
    carrier_stats = df.groupby('carrier')['delay_hours'].agg(['mean', 'count']).reset_index()
    carrier_stats.columns = ['carrier', 'carrier_avg_delay', 'carrier_volume']
    
    # Carrier Bias: Avg difference between Actual and ETA. 
    df['carrier_bias_hours'] = df['carrier'].map(carrier_stats.set_index('carrier')['carrier_avg_delay'])
    
    # On Time Ratio (Tolerance: < 2 hours)
    df['is_on_time'] = (df['delay_hours'] < 2).astype(int)
    carrier_otr = df.groupby('carrier')['is_on_time'].mean().reset_index(name='carrier_ontime_ratio')
    
    df = df.merge(carrier_stats.drop(columns=['carrier_avg_delay']), on='carrier', how='left') # Don't duplicate avg
    df = df.merge(carrier_otr, on='carrier', how='left') # Merge Ratio
    
    # Carrier Avg Delay (Global)
    df['carrier_avg_delay'] = df['carrier_bias_hours'] 
    
    # C. Time-Based Features
    print("Generating Time Features...")
    df['day_of_week'] = df['departure_time'].dt.dayofweek # 0=Mon, 6=Sun
    df['month'] = df['departure_time'].dt.month
    # Peak Season: Nov(11), Dec(12). Maybe Oct(10)? User said "Nov-Dec"
    df['is_peak_season'] = df['month'].isin([11, 12]).astype(int)
    
    # D. Congestion & Risk Indicators
    # Origin/Dest Delay Index (Avg delay at port)
    print("Generating Risk Scores...")
    origin_delay = df.groupby('origin')['delay_hours'].mean().reset_index(name='origin_port_delay_index')
    dest_delay = df.groupby('destination')['delay_hours'].mean().reset_index(name='destination_port_delay_index')
    
    # NEW: Via Port Congestion (Multi-Hop Support)
    # Explode the pipe-separated via_ports to calculate stats per individual port
    print("Calculating Transshipment Hub Risks...")
    
    # 1. Create a temporary dataframe of individual stops
    stops = df[['via_port', 'delay_hours']].copy()
    stops['via_port'] = stops['via_port'].astype(str).str.split('|')
    stops = stops.explode('via_port')
    stops = stops[stops['via_port'] != 'DIRECT'] # Filter out direct
    
    # 2. Calculate risk index per unique hub
    hub_risk = stops.groupby('via_port')['delay_hours'].mean().reset_index(name='hub_risk_index')
    hub_risk_map = hub_risk.set_index('via_port')['hub_risk_index'].to_dict()
    
    # 3. Map back to main DF: Average risk of all stops in the route
    def get_route_risk(via_str):
        if not via_str or via_str == 'DIRECT':
            return 0
        ports = str(via_str).split('|')
        risks = [hub_risk_map.get(p, 0) for p in ports]
        return np.mean(risks) if risks else 0

    df['via_port_congestion_index'] = df['via_port'].apply(get_route_risk)
    
    df = df.merge(origin_delay, on='origin', how='left')
    df = df.merge(dest_delay, on='destination', how='left')
    # df = df.merge(via_delay, on='via_port', how='left') -> REMOVED simple merge using helper instead
    
    # Route Risk Score
    # Composite of Lane Volatility + Congestion + Via Port Risk
    
    # Calculate Throughput first (needed for saving)
    origin_vol = df.groupby('origin')['shipment_id'].count().reset_index(name='origin_throughput_score')
    dest_vol = df.groupby('destination')['shipment_id'].count().reset_index(name='destination_throughput_score')
    
    # Normalize
    max_vol = df['lane_delay_volatility'].max()
    df['route_risk_score'] = (
        (df['lane_delay_volatility'] / (max_vol + 0.1)) * 40 + 
        (df['destination_port_delay_index'].clip(0, 100) / 100) * 40 +
        (df['via_port_congestion_index'].clip(0, 100) / 100) * 20
    ).fillna(0)
    
    # Weather Flag (Mock)
    # 5% random
    df['weather_flag'] = (np.random.rand(len(df)) < 0.05).astype(int)
    
    # --- "HACKATHON GOLD" FEATURES ---
    
    # Prediction Confidence (0.0 - 1.0)
    # Logic: High Volume + Low Volatility + High Carrier Consistency = High Confidence
    print("Calculating Confidence Scores...")
    
    # Normalize Metrics 0-1
    # Volume: Log scale to dampen huge lanes
    vol_score = np.log1p(df['lane_volume']) / np.log1p(df['lane_volume'].max())
    
    # Volatility: Inverse (Low vol = high score)
    # Cap volatility at reasonable hours to avoid skew
    vol_capped = df['lane_delay_volatility'].clip(0, 48) 
    vol_score = 1 - (vol_capped / 48)
    
    # Carrier Consistency: on_time_ratio
    carrier_score = df['carrier_ontime_ratio']
    
    # Weighted Average
    # 40% Lane Stability (Vol), 30% Data Support (Vol), 30% Carrier
    df['prediction_confidence'] = (vol_score * 0.4 + vol_score * 0.3 + carrier_score * 0.3).clip(0, 1)
    
    
    # --- SAVE FEATURE STORE (ARTIFACTS) FOR INFERENCE ---
    import pickle
    print("Saving Feature Store Artifacts...")
    
    # Merge lane stats for the store
    lane_full_stats = lane_stats.merge(lane_transit, on='lane', how='left')
    
    # Merge carrier stats for the store
    carrier_full_stats = carrier_stats.merge(carrier_otr, on='carrier', how='left')
    carrier_full_stats['carrier_reliability_score'] = (carrier_full_stats['carrier_ontime_ratio'] * 100).clip(0, 100)
    
    feature_store = {
        'lane_stats': lane_full_stats.set_index('lane')[['lane_avg_delay', 'lane_delay_volatility', 'lane_volume', 'lane_avg_transit_hours']].to_dict('index'),
        'carrier_stats': carrier_full_stats.set_index('carrier')[['carrier_volume', 'carrier_ontime_ratio', 'carrier_reliability_score']].to_dict('index'),
        'carrier_bias': df.groupby('carrier')['carrier_bias_hours'].mean().to_dict(),
        'origin_stats': origin_delay.set_index('origin')['origin_port_delay_index'].to_dict(),
        'dest_stats': dest_delay.set_index('destination')['destination_port_delay_index'].to_dict(),
        'via_stats': hub_risk_map, # DICT of Port -> Score
        'origin_vol': origin_vol.set_index('origin')['origin_throughput_score'].to_dict(),
        'dest_vol': dest_vol.set_index('destination')['destination_throughput_score'].to_dict(),
        'max_vol': float(df['lane_volume'].max()), # For normalization
    }
    
    with open('feature_store.pkl', 'wb') as f:
        pickle.dump(feature_store, f)
        
    print("Feature Store saved to feature_store.pkl")

    # --- FINAL COLUMN SELECTION & ORDERING ---
    columns_ordered = [
        # 1. Identity & Raw
        'shipment_id', 'origin', 'destination', 'via_port', 'carrier', 'service_type',
        'departure_time', 'carrier_eta', 'actual_arrival',
        
        # 2. Target
        'delay_hours',
        
        # 3. Engineered - Lane
        'lane', 'lane_avg_delay', 'lane_delay_volatility', 'lane_avg_transit_hours',
        'lane_volume', 
        
        # 4. Engineered - Carrier
        'carrier_avg_delay', 'carrier_ontime_ratio', 'carrier_bias_hours',
        
        # 5. Engineered - Time
        'day_of_week', 'month', 'is_peak_season',
        
        # 6. Engineered - Risk/Context
        'origin_port_delay_index', 'destination_port_delay_index', 'via_port_congestion_index',
        'route_risk_score', 'weather_flag',
        
        # 7. Gold / Meta
        'prediction_confidence'
    ]
    
    # Ensure no missing cols
    for c in columns_ordered:
        if c not in df.columns:
            print(f"Warning: Column {c} missing. Creating empty.")
            df[c] = 0
            
    final_df = df[columns_ordered]
    
    print(f"Saving {len(final_df)} rows to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("Success! Hackathon Dataset Ready.")

if __name__ == "__main__":
    generate_features()
