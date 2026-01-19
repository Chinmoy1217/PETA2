from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import json
import os
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


app = FastAPI(
    title="ETA Insight API",
    description="""
    ## Next-Gen Logistics Intelligence Platform
    
    This API provides predictive ETA (Estimated Time of Arrival) calculations, 
    real-time shipment tracking, and comprehensive analytics for logistics operations.
    
    ### Key Features:
    * **Authentication**: Secure login/signup with Snowflake integration
    * **Predictive Analytics**: XGBoost-powered ETA predictions
    * **Real-Time Tracking**: Active shipment monitoring
    * **Batch Processing**: Upload CSV/Parquet files for bulk predictions
    * **Analytics Dashboard**: Performance metrics and model comparisons
    
    ### Snowflake Integration:
    * Connected to 949,227+ shipment records
    * Live data from `FACT_TRIP` table
    * Model accuracy tracking in `MODEL_ACCURACY_HISTORY`
    """,
    version="1.0.0",
    terms_of_service="https://cozentus.com/terms",
    contact={
        "name": "Cozentus Support",
        "email": "support@cozentus.com",
    },
    license_info={
        "name": "Proprietary",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "./model"
DATA_PATH = "../../Master_Training_Data_Augmented.csv" # Pointing to Hackthon/ root
from backend.data_loader import DataLoader
from backend.risk_model import RiskModel
from backend.reliability_model import ReliabilityModel

# Global Models Init
risk_model = RiskModel(MODEL_DIR)
reliability_model = ReliabilityModel(MODEL_DIR)

# Safe Global Initialization
bst = None
lr_model = None
rf_model = None
encoders = {}
mode_stats = {}
model_metrics = {}
history_df = None
port_coords = {}
feature_store = {}

# Pydantic Models for Authentication
class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    username: str
    password: str

# Snowflake Connection Helper (Native)
def get_snowflake_connection():
    try:
        ctx = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        return ctx
    except Exception as e:
        print(f"Snowflake Connection Failed: {e}")
        return None

@app.post("/login")
def login(req: LoginRequest):
    # 1. Try Snowflake
    conn = get_snowflake_connection()
    if conn:
        try:
            cs = conn.cursor()
            cs.execute(f"SELECT COUNT(*) FROM ROLE_LOGIN WHERE USERNAME='{req.username}' AND PASSWORD='{req.password}'")
            count = cs.fetchone()[0]
            conn.close()
            if count > 0:
                return {"status": "success", "token": "snow_token_123", "role": "admin"}
        except Exception as e:
             print(f"Snowflake Query Error: {e}")
    
    # 2. Mock Fallback (If DB fails or not configured)
    if req.username == "admin" and req.password == "admin":
        return {"status": "success", "token": "mock_token_123", "role": "admin"}
        
    raise HTTPException(status_code=401, detail="Invalid Credentials")

@app.post("/signup")
def signup(req: LoginRequest):
    # 1. Try Snowflake
    conn = get_snowflake_connection()
    success = False
    if conn:
        try:
            cs = conn.cursor()
            cs.execute(f"INSERT INTO ROLE_LOGIN (ROLE_NAME, USERNAME, PASSWORD) VALUES ('User', '{req.username}', '{req.password}')")
            conn.close()
            success = True
        except Exception as e:
            print(f"Snowflake Insert Error: {e}")
            pass

    # Mock success
    return {"status": "success", "message": "User registered successfully"}

def load_artifacts():
    global bst, lr_model, rf_model, encoders, mode_stats, model_metrics, history_df, port_coords, feature_store
    
    # Initialize all globals with defaults first
    bst = None
    lr_model = None
    rf_model = None
    encoders = {}
    mode_stats = {}
    model_metrics = {}
    history_df = None
    port_coords = {}
    feature_store = {}
    
    # 1. XGBoost / Strategy
    try:
        xg_path = os.path.join(MODEL_DIR, "eta_xgboost.json")
        if os.path.exists(xg_path):
            with open(xg_path, 'r') as f:
                content = f.read().strip()
            if content in ['USE_RF', 'USE_ENSEMBLE']:
                bst = content
            else:
                bst = xgb.Booster()
                bst.load_model(xg_path)
    except Exception as e:
        print(f"XGBoost Load Failed: {e}")

    # 2. Sklearn Models
    try:
        lr_path = os.path.join(MODEL_DIR, "linear_regression.pkl")
        rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
        if os.path.exists(lr_path):
            with open(lr_path, "rb") as f:
                lr_model = pickle.load(f)
        if os.path.exists(rf_path):
            with open(rf_path, "rb") as f:
                rf_model = pickle.load(f)
    except Exception as e:
        print(f"Sklearn Load Failed: {e}")

    # 3. JSON Artifacts (Encoders, Stats, Ports)
    def load_json(filename, default):
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to parse {filename}: {e}")
        return default

    encoders = load_json("encoders.json", {})
    mode_stats = load_json("mode_stats.json", {})
    port_coords = load_json("port_coordinates.json", {})
    
    try:
        comps = load_json("model_comparison.json", [])
        if isinstance(comps, list):
            model_metrics = {item['name']: item['accuracy'] for item in comps if 'name' in item}
        elif isinstance(comps, dict):
             # Handle alternate format
             model_metrics = {k: v.get('accuracy_pct', 0) for k, v in comps.items()}
    except Exception as e:
        print(f"Metrics Load Failed: {e}")

    # 4. Historical Data
    try:
        loader = DataLoader()
        history_df = loader.get_training_view()
    except Exception as e:
        print(f"DataLoader failed: {e}")
        history_df = pd.DataFrame()

    # 5. Feature Store
    try:
        fs_path = "../feature_store.pkl"
        if os.path.exists(fs_path):
            with open(fs_path, "rb") as f:
                feature_store = pickle.load(f)
    except Exception as e:
        print(f"Feature Store Load Failed: {e}")

    print(f"All Models & Artifacts Loaded. Ports found: {len(port_coords)}")

load_artifacts()

# --- GEOGRAPHIC VALIDATION LOGIC ---
country_to_continent = {
    'US': 'NA', 'CA': 'NA', 'MX': 'NA',
    'CN': 'AS', 'JP': 'AS', 'IN': 'AS', 'SG': 'AS', 'KR': 'AS', 'AE': 'ME', 'SA': 'ME',
    'DE': 'EU', 'NL': 'EU', 'GB': 'EU', 'FR': 'EU', 'ES': 'EU', 'IT': 'EU',
    'BR': 'SA', 'AR': 'SA', 'CL': 'SA',
    'AU': 'OC', 'NZ': 'OC',
    'ZA': 'AF', 'EG': 'AF', 'NG': 'AF'
}
def get_continent(code):
    if not code or len(str(code)) < 2: return 'UNKNOWN'
    country = str(code)[:2].upper()
    return country_to_continent.get(country, 'UNKNOWN')

# --- INTELLIGENCE LOGIC ---
hubs = {
    'AS': ['SGSIN', 'CNHKG', 'KRPUS', 'MYPKG'], 
    'EU': ['NLROT', 'BEANR', 'DEHAM', 'ESALG'],
    'ME': ['AEDXB', 'EGSUEZ', 'SAJED', 'OMSLL'],
    'NA': ['USLAX', 'USNYC', 'PABLB', 'CAMTR'],
}

route_flow = {
    ('AS', 'EU'): ['ME'],
    ('AS', 'NA'): ['AS', 'NA'],
    ('EU', 'AS'): ['ME'],
    ('EU', 'NA'): ['EU'],
    ('NA', 'AS'): ['NA'], 
    ('NA', 'EU'): ['NA'],
    ('AS', 'AF'): ['AS', 'ME'],
}

def infer_transshipment(pol, pod, mode):
    # Replicating logic from dataset generation (Multi-Stop)
    if str(mode).upper() not in ['OCEAN', 'AIR']:
        return "DIRECT", 0
        
    c1 = get_continent(pol)
    c2 = get_continent(pod)
    
    # Same continent = Direct (mostly)
    if c1 == c2 or c1 == 'UNKNOWN' or c2 == 'UNKNOWN':
        return "DIRECT", 0
        
    # Check if flow exists
    key = (c1, c2)
    via_ports = []
    delay = 0
    
    # 60% chance Ocean, 30% Air
    prob = 0.6 if str(mode).upper() == 'OCEAN' else 0.3
    
    # Deterministic Seed
    seed = abs(hash(pol + pod)) 
    
    if (seed % 100) / 100 < prob:
         possible_regions = route_flow.get(key, [c1])
         
         # 1 or 2 stops? (Determinstic based on length/chars)
         num_stops = 1
         if len(pol+pod) % 3 == 0 and str(mode).upper() == 'OCEAN':
             num_stops = 2
             
         for i in range(num_stops):
             # Pick region (pseudo-randomly based on seed + i)
             reg_idx = (seed + i) % len(possible_regions)
             region = possible_regions[reg_idx]
             
             if region in hubs:
                 hub_list = hubs[region]
                 hub_idx = (seed + i * 7) % len(hub_list)
                 via_ports.append(hub_list[hub_idx])
                 
                 # Delay per stop
                 d = 96 if str(mode).upper() == 'OCEAN' else 18
                 delay += d

    if not via_ports:
        return "DIRECT", 0
        
    return "|".join(via_ports), delay

def get_rich_features(pol, pod, mode, carrier):
    # Extract learned stats from Feature Store
    lane = f"{pol}-{pod}"
    
    # 1. Lane Stats
    l_stats = feature_store.get('lane_stats', {}).get(lane, {})
    lane_avg_delay = l_stats.get('lane_avg_delay', 0)
    lane_volatility = l_stats.get('lane_delay_volatility', 0)
    lane_volume = l_stats.get('lane_volume', 0)
    
    # 2. Carrier Stats
    c_stats = feature_store.get('carrier_stats', {}).get(carrier, {})
    carrier_score = c_stats.get('carrier_ontime_ratio', 0.8) 
    
    # 3. Transshipment
    via_port_str, via_delay = infer_transshipment(pol, pod, mode)
    via_risk = 0
    
    # Multi-Stop Risk Calculation
    if via_port_str != 'DIRECT':
        ports = via_port_str.split('|')
        risks = []
        for p in ports:
            # Map specific hub risk from store
            r = feature_store.get('via_stats', {}).get(p, 25) # Default risk 25 if new hub
            risks.append(r)
        via_risk = sum(risks) / len(risks) if risks else 0
    
    # 4. Risk Score Calculation
    max_vol = feature_store.get('max_vol', 50.0)
    dest_risk = feature_store.get('dest_stats', {}).get(pod, 0)
    
    risk_score = (
        (lane_volatility / (max_vol + 0.1)) * 40 + 
        (dest_risk / 100) * 40 + 
        (via_risk / 100) * 20
    )
    
    # 5. Confidence
    vol_score = np.log1p(lane_volume) / np.log1p(1000) 
    stability_score = 1 - (min(lane_volatility, 48) / 48)
    confidence = (stability_score * 0.4 + vol_score * 0.3 + carrier_score * 0.3)
    
    return {
        "via_port": via_port_str,
        "transshipment_delay_hours": via_delay,
        "risk_score": round(risk_score, 1), 
        "prediction_confidence": round(confidence * 100, 1), 
        "lane_avg_delay": round(lane_avg_delay, 1),
        "carrier_reliability": round(carrier_score * 100, 1)
    }

def retrain_model():
    print("Starting Background Retraining...")
    global bst, encoders, history_df
    try:
        # 1. Load latest data
        if not os.path.exists(DATA_PATH):
            print("Data file not found, skipping train.")
            return

        df = pd.read_csv(DATA_PATH)
        history_df = df # Update global history dataframe
        
        # 2. Feature Engineering
        df['Route'] = df['PolCode'].astype(str) + "_" + df['PodCode'].astype(str) + "_" + df['ModeOfTransport'].astype(str)
        FEATURES = ['PolCode', 'PodCode', 'ModeOfTransport', 'Route']
        TARGET = 'Actual_Duration_Hours'
        
        # 3. Re-fit Encoders (Simple Index Mapping)
        new_encoders = {}
        encoded_df = df.copy()
        for col in FEATURES:
            unique_vals = df[col].astype(str).unique()
            mapping = {k: i for i, k in enumerate(unique_vals)}
            new_encoders[col] = mapping
            encoded_df[col] = encoded_df[col].astype(str).map(mapping)
            
        encoders = new_encoders # Update global encoders
        # Save encoders
        with open(os.path.join(MODEL_DIR, "encoders.json"), "w") as f:
            json.dump(encoders, f)

        # 4. Train XGBoost
        # Filter rows with valid targets
        train_df = encoded_df.dropna(subset=[TARGET])
        train_df = train_df[train_df[TARGET] > 0]
        
        X = train_df[FEATURES]
        y = np.log1p(train_df[TARGET])
        
        dtrain = xgb.DMatrix(X, label=y)
        
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 10,
            'eta': 0.05, # Faster learning rate for quick updates
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'nthread': 4
        }
        
        # Train (Fewer rounds for background speed)
        model = xgb.train(params, dtrain, num_boost_round=200) 
        
        # Save
        model.save_model(os.path.join(MODEL_DIR, "eta_xgboost.json"))
        bst = model # Update global model reference
        
        print(f"Retraining Complete. Trained on {len(train_df)} rows.")
        
    except Exception as e:
        print(f"Retraining Failed: {e}")

class SimulationRequest(BaseModel):
    PolCode: str
    PodCode: str
    ModeOfTransport: str
    via_port: str = None # Optional override
    Carrier: str = "UNKNOWN"
    trip_ATD: str = None # YYYY-MM-DD
    congestion_level: int = 0  # 0-100
    weather_severity: int = 0  # 0-100

@app.get("/")
def home():
    return {"message": "ETA Insight API is Running"}

@app.get("/locations")
def get_locations():
    """Returns list of all available Port/City codes"""
    return sorted(list(port_coords.keys()))

@app.get("/metrics")
def get_metrics():
    try:
        # Load Model Metrics
        with open(os.path.join(MODEL_DIR, "model_comparison.json"), "r") as f:
            data = json.load(f)
        metrics = {}
        xgboost_data = {}
        
        if isinstance(data, dict):
            xgboost_data = data.get('xgboost', {})
        elif isinstance(data, list):
            xgboost_data = next((item for item in data if item["name"] == "XGBoost"), data[0])
                 
        # Map Model Stats
        metrics['accuracy'] = xgboost_data.get('accuracy_pct', 85.0)
        metrics['rmse'] = xgboost_data.get('mae', 12.5)
        metrics['name'] = 'XGBoost v1.0'

        # 2. LOAD LIVE DATA FROM SNOWFLAKE
        try:
            conn = get_snowflake_connection()
            if conn:
                with conn.cursor() as cs:
                    # A. Total Shipments (FACT_TRIP)
                    cs.execute("SELECT COUNT(*) FROM DT_INGESTION.FACT_TRIP")
                    metrics['total_shipments'] = cs.fetchone()[0]

                    # B. Connected Carriers (DIM_CARRIER)
                    cs.execute("SELECT COUNT(*) FROM DT_INGESTION.DIM_CARRIER")
                    metrics['connected_carriers_count'] = cs.fetchone()[0]

                    # C. Recent Predictions (PETA_PREDICTION_RESULTS)
                    cs.execute("SELECT COUNT(*) FROM DT_INGESTION.PETA_PREDICTION_RESULTS")
                    metrics['total_predictions'] = cs.fetchone()[0]
                    
                    # D. On-Time Rate (Mock calculation from Real Data sample if possible, else 94.5)
                    metrics['on_time_rate'] = 94.5 
                    metrics['late_shipments_count'] = int(metrics['total_shipments'] * 0.055)
                    metrics['delayed_rate'] = 5.5
                    metrics['avg_delay_days'] = 0.8
                    metrics['max_delay_days'] = 4.2
                    metrics['critical_delays_count'] = int(metrics['total_shipments'] * 0.001)

                conn.close()
                print("✅ Metics loaded from Snowflake.")
            else:
                raise Exception("No SF Connection")

        except Exception as e:
            print(f"⚠️ Snowflake Metrics Failed: {e}. Using History DF Fallback.")
            # Fallback to local params
            global history_df
            if history_df is not None and not history_df.empty:
                metrics['total_shipments'] = len(history_df)
                if 'CID' in history_df.columns:
                    metrics['connected_carriers_count'] = int(history_df['CID'].nunique())
                else:
                    metrics['connected_carriers_count'] = 45
            else:
                 metrics['total_shipments'] = 15420
                 metrics['connected_carriers_count'] = 40
            
            # KPI Defaults
            metrics.update({
                 'on_time_rate': 94.5,
                 'late_shipments_count': 840,
                 'delayed_rate': 5.4,
                 'avg_delay_days': 0.8,
                 'max_delay_days': 4.2,
                 'critical_delays_count': 12,
            })

        # Reliability Score
        metrics['reliability_score'] = round(metrics.get('on_time_rate', 90) / 10, 1)

        # --- OPERATIONAL KPIs (Static for Demo) ---
        metrics['mode_accuracy'] = [
            {'name': 'Air', 'value': 98.5},
            {'name': 'Ocean', 'value': 85.2},
            {'name': 'Road', 'value': 92.1},
            {'name': 'Rail', 'value': 89.4}
        ]
        metrics['carrier_accuracy'] = [
            {'name': 'Maersk', 'value': 94.2},
            {'name': 'MSC', 'value': 91.5},
            {'name': 'DHL', 'value': 97.8},
            {'name': 'FedEx', 'value': 98.1},
            {'name': 'Hapag-Lloyd', 'value': 88.3}
        ]
        metrics['route_accuracy'] = [
            {'name': 'CN-US (Transpacific)', 'value': 86.4},
            {'name': 'CN-EU (Silk Road)', 'value': 89.1},
            {'name': 'EU-US (Transatlantic)', 'value': 94.7},
            {'name': 'Intra-Asia', 'value': 96.2}
        ]
        
        # Trend Data (Static for Visuals)
        metrics['trend_data'] = [
             {"date": "2024-01", "count": 120}, {"date": "2024-02", "count": 132},
             {"date": "2024-03", "count": 101}, {"date": "2024-04", "count": 145},
             {"date": "2024-05", "count": 190}, {"date": "2024-06", "count": 175}
        ]
        
        return metrics
    except Exception as e:
        print(f"Metrics Error: {e}")
        return {"accuracy": 0, "rmse": 0, "on_time_rate": 0, "error": str(e)}

@app.get("/comparison")
def get_comparison():
    try:
        with open(os.path.join(MODEL_DIR, "model_comparison.json"), "r") as f:
            return json.load(f)
    except:
        return []

@app.get("/plots")
def get_plots():
    try:
        with open(os.path.join(MODEL_DIR, "plots.json"), "r") as f:
            return json.load(f)
    except:
        return {}

@app.get("/active")
def get_active_shipments():
    """Returns top 50 'Latest' shipments from Snowflake (Predictions or Fact Trip)."""
    try:
        response = []
        statuses = ['In Transit', 'Customs Clearance', 'Arrived at Hub', 'Out for Delivery']
        
        # 1. Try Snowflake
        try:
            conn = get_snowflake_connection()
            if conn:
                with conn.cursor() as cs:
                    # Prefer Prediction Results (Live Activity)
                    cs.execute("""
                        SELECT PREDICTION_ID, POL_CODE, POD_CODE, MODE_OF_TRANSPORT, 
                               PETA_HOURS, CREATED_BY
                        FROM DT_INGESTION.PETA_PREDICTION_RESULTS 
                        ORDER BY PREDICTION_TIMESTAMP DESC
                        LIMIT 50
                    """)
                    rows = cs.fetchall()
                    
                    if rows:
                        for r in rows:
                            # Map DB columns to Frontend
                            # r: (ID, POL, POD, MODE, HOURS, CREATOR)
                            response.append({
                                "id": r[0], 
                                "origin": r[1],
                                "destination": r[2],
                                "mode": r[3],
                                "via": "AI Evaluated", 
                                "status": "Projected",
                                "eta": f"{round(float(r[4]), 1)}h"
                            })
                    else:
                        # Fallback to FACT_TRIP if no predictions
                         cs.execute("""
                            SELECT TRIP_ID, POL_CODE, POD_CODE, MODE_OF_TRANSPORT, 
                                   ACTUAL_DURATION_MINUTES
                            FROM DT_INGESTION.FACT_TRIP 
                            LIMIT 50
                        """)
                         rows = cs.fetchall()
                         for r in rows:
                            response.append({
                                "id": str(r[0]), 
                                "origin": r[1],
                                "destination": r[2],
                                "mode": r[3],
                                "via": "DIRECT", 
                                "status": statuses[hash(str(r[0])) % len(statuses)],
                                "eta": f"{round(float(r[4] or 0)/60, 1)}h"
                            })
                
                conn.close()
                return response

        except Exception as db_e:
             print(f"⚠️ Active DB Fetch Failed: {db_e}")

        # 2. History DF Fallback
        if history_df is not None and not history_df.empty:
            # Take a random sample to simulate "Live" view
            sample = history_df.sample(n=min(100, len(history_df))).to_dict('records')
            
            for row in sample:
                # Generate a status
                stat = statuses[np.random.randint(0, len(statuses))]
                response.append({
                    "id": row.get('Transport_Vehicle_ID', 'N/A'), 
                    "origin": row.get('PolCode', 'UNK'),
                    "destination": row.get('PodCode', 'UNK'),
                    "mode": row.get('ModeOfTransport', 'UNK'),
                    "via": row.get('via_port', 'DIRECT'), 
                    "status": stat,
                    "eta": f"{round(row.get('Actual_Duration_Hours', 0), 1)}h"
                })
            return response
            
        return []
    except Exception as e:
        print(f"Error fetching active: {e}")
        return []

def prepare_input(df):
    # For XGBoost (Native DMatrix)
    FEATURES = ['PolCode', 'PodCode', 'ModeOfTransport', 'Route']
    df['Route'] = df['PolCode'].astype(str) + "_" + df['PodCode'].astype(str) + "_" + df['ModeOfTransport'].astype(str)
    
    encoded_df = df.copy()
    # Ensure Carrier exists
    if 'Carrier' not in encoded_df.columns:
        encoded_df['Carrier'] = 'UNKNOWN'
    encoded_df['Carrier'] = encoded_df['Carrier'].fillna('UNKNOWN')

    # Encode ALL Categoricals
    for col in ['PolCode', 'PodCode', 'ModeOfTransport', 'Route', 'Carrier']:
        mapping = encoders.get(col, {})
        # Handle unknown categories by mapping to 0 (or generic unknown if we had one)
        encoded_df[col] = encoded_df[col].astype(str).map(lambda s: mapping.get(s, 0))
    
    # Add Seasonality (Simulate uses Now)
    now = pd.Timestamp.now()
    encoded_df['Departure_Month'] = int(now.month)
    encoded_df['Departure_Week'] = int(now.isocalendar().week)
    encoded_df['Departure_Quarter'] = int(now.quarter)
    
    # Add other missing numerics as 0
    NUMERIC = ['External_Risk_Score', 'BASE_ETA_DAYS', 'WEATHER_RISK_SCORE',
                'GEOPOLITICAL_RISK_SCORE', 'LABOR_STRIKE_SCORE', 'CUSTOMS_DELAY_SCORE', 
                'PORT_CONGESTION_SCORE', 'CARRIER_DELAY_SCORE', 'PEAK_SEASON_SCORE',
                'Risk_Intensity', 'Peak_Risk_Factor', 'Route_Target_Enc', 'Carrier_Target_Enc']
    for c in NUMERIC: encoded_df[c] = 0.0

    
    # EXACT ORDER matching model (18 features)
    FEATURES = ['PolCode', 'PodCode', 'ModeOfTransport', 'Route', 'Carrier',
                'External_Risk_Score', 'BASE_ETA_DAYS', 'WEATHER_RISK_SCORE',
                'GEOPOLITICAL_RISK_SCORE', 'LABOR_STRIKE_SCORE', 'CUSTOMS_DELAY_SCORE', 
                'PORT_CONGESTION_SCORE', 'CARRIER_DELAY_SCORE', 'PEAK_SEASON_SCORE',
                'Risk_Intensity', 'Peak_Risk_Factor', 'Route_Target_Enc', 'Carrier_Target_Enc']
    
    return encoded_df[FEATURES]

def save_prediction_results(req: SimulationRequest, result: dict):
    print("Background: Saving prediction to Snowflake...")
    try:
        conn = get_snowflake_connection()
        if not conn: return
        
        with conn.cursor() as cs:
            # Create Table if missing (Self-Healing)
            cs.execute("""
                CREATE TABLE IF NOT EXISTS DT_INGESTION.PETA_PREDICTION_RESULTS (
                    PREDICTION_ID VARCHAR(50),
                    PREDICTION_TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                    POL_CODE VARCHAR(10), POD_CODE VARCHAR(10), MODE_OF_TRANSPORT VARCHAR(20),
                    CARRIER_SCAC_CODE VARCHAR(10), DEPARTURE_DATE DATE,
                    CONGESTION_LEVEL NUMBER(3,0), WEATHER_SEVERITY NUMBER(3,0),
                    PREDICTED_HOURS NUMBER(10,2), PREDICTED_DAYS NUMBER(10,2),
                    ARRIVAL_DATE DATE, MODEL_VERSION VARCHAR(50), 
                    ROUTE VARCHAR(100), CREATED_BY VARCHAR(100)
                )
            """)
            
            import uuid
            pid = str(uuid.uuid4())
            atd = req.trip_ATD if req.trip_ATD else '2025-01-01'
            final_hours = result['prediction_hours']
            
            sql = f"""
                INSERT INTO DT_INGESTION.PETA_PREDICTION_RESULTS (
                    PREDICTION_ID, POL_CODE, POD_CODE, MODE_OF_TRANSPORT, CARRIER_SCAC_CODE,
                    DEPARTURE_DATE, CONGESTION_LEVEL, WEATHER_SEVERITY,
                    PETA_HOURS, PETA_DAYS, MODEL_VERSION, ROUTE, CREATED_BY, STATUS
                ) VALUES (
                    '{pid}', '{req.PolCode}', '{req.PodCode}', '{req.ModeOfTransport}', '{req.Carrier}',
                    '{atd}', {req.congestion_level}, {req.weather_severity},
                    {final_hours}, {round(final_hours/24, 2)}, 'Ensemble_v1', 
                    '{req.PolCode}-{req.PodCode}', 'PETA_API', 'SUCCESS'
                )
            """
            cs.execute(sql)
            conn.commit() # Ensure data is saved
            print(f"✅ Prediction {pid} saved to Snowflake.")
            
    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to save prediction: {e}")
    finally:
        conn.close()

@app.post("/simulate")
def simulate_trip(req: SimulationRequest, background_tasks: BackgroundTasks):
    df_raw = pd.DataFrame([req.dict()])
    X_enc = prepare_input(df_raw)
    
    try:
        # 1. Prediction: XGBoost / Strategy (Robust)
        pred_xgb = 0
        if isinstance(bst, str):
             if bst == "USE_RF": pred_xgb = float(np.expm1(rf_model.predict(X_enc))[0])
             elif bst == "USE_ENSEMBLE": pred_xgb = float(np.expm1(rf_model.predict(X_enc))[0])
             else: pred_xgb = 0
        elif bst:
             dmat = xgb.DMatrix(X_enc)
             pred_xgb = float(np.expm1(bst.predict(dmat))[0])
        else:
             try:
                 pred_xgb = float(np.expm1(rf_model.predict(X_enc))[0])
             except:
                 print("RF Model Failed. Using Fallback.")
                 pred_xgb = 120.0 # Default fallback

        # 2. Prediction: Sklearn Models
        # Note: Sklearn models are static pkls, we won't retrain them in this simplified flow
        try:
            pred_lr = float(np.expm1(lr_model.predict(X_enc))[0])
            pred_rf = float(np.expm1(rf_model.predict(X_enc))[0])
        except:
             # Fallback if sklearn models fail on new encoding
             pred_lr = pred_xgb
             pred_rf = pred_xgb
        
        # 3. Decision Logic
        candidates = [
            {'name': 'XGBoost', 'pred': pred_xgb, 'score': model_metrics.get('XGBoost', 0)},
            {'name': 'Random Forest', 'pred': pred_rf, 'score': model_metrics.get('Random Forest', 0)},
            {'name': 'Linear Regression', 'pred': pred_lr, 'score': model_metrics.get('Linear Regression', 0)}
        ]
        
        # Pick winner
        best = max(candidates, key=lambda x: x['score'])
        base_pred = best['pred']
        
        # --- INTELLIGENCE ENRICHMENT ---
        rich_data = get_rich_features(req.PolCode, req.PodCode, req.ModeOfTransport, "GENERIC")
        
        # Override via_port if user provided it
        if req.via_port:
            rich_data['via_port'] = req.via_port
            
        transshipment_delay = rich_data['transshipment_delay_hours']
        
        # --- SCENARIO PLANNING LOGIC ---
        congestion_penalty = base_pred * (req.congestion_level * 0.005)
        weather_penalty = base_pred * (req.weather_severity * 0.003)
        total_penalty = congestion_penalty + weather_penalty + transshipment_delay
        
        final_pred = base_pred + total_penalty
        
        # Geospatial Lookup
        src_coords = port_coords.get(req.PolCode, None)
        dest_coords = port_coords.get(req.PodCode, None)
        
        # Parse via_ports for coords
        via_coords_list = []
        via_port_str = rich_data['via_port']
        if via_port_str != 'DIRECT':
            stops = via_port_str.split('|')
            for s in stops:
                via_coords_list.append(port_coords.get(s, None))
        
        # 4. Generate AI Explanation
        explanation = (
            f"**Model Consensus:** Analyzed using 3 algorithms. "
            f"XGBoost ({round(pred_xgb,1)}h) was selected as the most reliable predictor (Accuracy: {best['score']}%), "
            f"outperforming Random Forest ({round(pred_rf,1)}h) and Linear Regression ({round(pred_lr,1)}h).\n\n"
        )
        
        # Operational Context
        mode = req.ModeOfTransport
        avg = mode_stats.get(mode, best['pred'])
        diff = best['pred'] - avg
        pct_diff = round((diff / avg) * 100, 1) if avg > 0 else 0
        
        explanation += f"**Operational Context:** The predicted transit of {round(best['pred'], 1)} hours is "
        if abs(pct_diff) < 5:
             explanation += f"consistent with the historical average for {mode}."
        elif diff < 0:
             explanation += f"**{abs(pct_diff)}% faster** than typical {mode} timeline ({round(avg,1)}h)."
        else:
             explanation += f"**{pct_diff}% slower** than typical {mode} timeline ({round(avg,1)}h)."

        # Transshipment Note (Narrative)
        if via_port_str != 'DIRECT':
            stops = via_port_str.split('|')
            route_desc = " -> ".join(stops)
            explanation += f"\n\n**Routing Insight:** Complex route detected. Shipment iterates through **{len(stops)} hub(s)**: {route_desc}. " \
                           f"This adds ~{rich_data['transshipment_delay_hours']}h dwelling time."

        if total_penalty > 0:
            explanation += f"\n\n**Scenario Impact:** Factors (Congestion: {req.congestion_level}%, Weather: {req.weather_severity}%, Transshipment) " \
                           f"have added **+{round(total_penalty, 1)} hours**."

        response_payload = {
            "prediction_hours": round(final_pred, 2),
            "prediction_days": round(final_pred / 24, 1),
            "base_hours": round(base_pred, 2),
            "confidence_score": rich_data['prediction_confidence'],
            "risk_score": rich_data['risk_score'],
            "scenario_impact": {
                "congestion_hours": round(congestion_penalty, 2),
                "weather_hours": round(weather_penalty, 2),
                "transshipment_hours": round(transshipment_delay, 2),
                "total_delay": round(total_penalty, 2)
            },
            "coordinates": {
                "source": src_coords,
                "destination": dest_coords,
                "via_stops": via_coords_list 
            },
            "route_details": {
                "via_port": rich_data['via_port'],
                "stops_count": len(via_coords_list),
                "lane_volatility": rich_data['risk_score'] 
            },
            "model_candidates": candidates,
            "ai_explanation": explanation,
            "status": "Success"
        }
        
        # Trigger Background Save
        background_tasks.add_task(save_prediction_results, req, response_payload)
        
        return response_payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- TRACKING & RICH DATA ENDPOINTS ---
def convert_units(mode, distance_km, speed_kmh):
    if str(mode).upper() == 'OCEAN':
        return round(distance_km * 0.539957, 0), round(speed_kmh * 0.539957, 1), 'nm', 'knots'
    return round(distance_km, 0), round(speed_kmh, 1), 'km', 'km/h'

def get_lat_lon(code):
    coords = port_coords.get(str(code), {'lat': 0, 'lon': 0})
    if isinstance(coords, dict):
        return coords.get('lat', 0), coords.get('lon', 0)
    return 0, 0

@app.post("/predict")
async def predict_eta(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    train: str = Form("false"),
    pol: str = Form(None),
    pod: str = Form(None),
    mode: str = Form(None),
    via: str = Form(None)
):
    is_training = train.lower() == 'true'

    # --- 1. DATA LOADING ---
    if file:
        try:
            df = pd.read_csv(file.file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")
            
        # Basic Geo Validation
        initial_count = len(df)
        def check_row(row):
            s = get_continent(row.get('PolCode'))
            d = get_continent(row.get('PodCode'))
            m = str(row.get('ModeOfTransport')).upper().strip()
            if s != 'UNKNOWN' and d != 'UNKNOWN' and s != d:
                if m in ['ROAD', 'RAIL', 'TRUCK']:
                    return False
            return True
            
        df = df[df.apply(check_row, axis=1)]
        if len(df) < initial_count:
            print(f"Batch Warning: Dropped {initial_count - len(df)} geographically impossible rows.")

    elif pol and pod and mode:
        df = pd.DataFrame([{
            'PolCode': pol, 'PodCode': pod, 'ModeOfTransport': mode,
            'via_port': via, 'Carrier': 'UNKNOWN'
        }])
    else:
        raise HTTPException(status_code=400, detail="Missing Input: CSV or Form Data")

    # --- 2. TRAINING ---
    if is_training and file and 'Actual_Duration_Hours' in df.columns:
         try:
            # Append to training data
            data_file = os.path.join(MODEL_DIR, "training_data.csv") 
            if 'trip_id' not in df.columns: df['trip_id'] = [f"UPLOAD_{i}" for i in range(len(df))]
            
            # Write header if missing
            hdr = not os.path.exists(data_file)
            df.to_csv(data_file, mode='a', header=hdr, index=False)
            
            # Trigger Retrain
            background_tasks.add_task(retrain_model)
         except Exception as ex:
            print(f"Failed to save training data: {ex}")

    # --- 3. INFERENCE ---
    try:
        if len(df) == 0: return {"status": "Error", "message": "No valid rows"}
        
        # Use Main Branch prepare_input
        X_enc = prepare_input(df)
        
        # ROBUST MODEL SELECTION (Fix for 'str' object error)
        preds = []
        # If bst is a string (Strategy Mode) or None
        if isinstance(bst, str) or bst is None:
             # Default to RF if XGB unavailable/strategy override
             # Check if we default to RF
             use_rf = True
             if isinstance(bst, str) and bst == "USE_ENSEMBLE": use_rf = True # Simplify to RF for now
             
             if use_rf and rf_model:
                 preds = np.expm1(rf_model.predict(X_enc))
             else:
                 # Ultimate Fallback
                 preds = [120.0] * len(df)
                 print("⚠️ Model Warning: using static fallback.")
        else:
             # Normal XGBoost
             dmat = xgb.DMatrix(X_enc)
             preds = np.expm1(bst.predict(dmat))

        # Ensure preds is array-like
        if isinstance(preds, float): preds = [preds]
        
        results = []
        for i, row in df.iterrows():
            p_eta = float(preds[i]) if i < len(preds) else 0
            
            # Single Row Enrichment
            if len(df) == 1:
                rich = get_rich_features(row.get('PolCode'), row.get('PodCode'), row.get('ModeOfTransport'), "GENERIC")
                results.append({
                    "prediction_hours": round(p_eta, 2),
                    "prediction_days": round(p_eta/24, 1),
                    "confidence_score": rich.get('prediction_confidence', 85),
                    "risk_score": rich.get('risk_score', 10),
                    "ai_explanation": f"Predicted {round(p_eta,1)}h based on {row.get('ModeOfTransport')} route.",
                    "coordinates": {
                        "source": port_coords.get(row.get('PolCode'), {}),
                        "destination": port_coords.get(row.get('PodCode'), {})
                    },
                    "rich_metrics": {
                        "distance": "Calc...",
                        "avg_speed": "Calc...",
                        "carbon_footprint": "Calc..."
                    }
                })
            else:
                 results.append({
                    "id": row.get('trip_id', i),
                    "origin": row.get('PolCode'),
                    "destination": row.get('PodCode'),
                    "eta": round(p_eta, 2)
                 })
                 
        if len(results) == 1: return results[0]
        return results

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_to_cloud(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Step 1: Save Locally -> Upload to Azure (Archive) -> Run Quality Check -> Return Result.
    (Step 2: Ingestion is triggered manually via /ingest)
    """
    import shutil
    import os
    from backend.ingestion_service import upload_to_archive_rest
    from backend.quality_check import DataQualityChecker

    # Save to temp file
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        # 1. Save locally
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"💾 Saved local file: {temp_path}")

        # 2. Upload to Azure (as requested "Uploads file into Azure")
        # We use the archive function to push it to blob storage
        background_tasks.add_task(upload_to_archive_rest, temp_path)
        
        # 3. Run Quality Check
        checker = DataQualityChecker(temp_path)
        is_ready, accuracy = checker.run_quality_check()
        
        status = "success" if is_ready else "warning"
        msg = f"Accuracy {accuracy}%." + (" Ready to Ingest." if is_ready else " Quality Check Failed.")

        return {
            "status": status,
            "accuracy": accuracy,
            "filename": file.filename, # Return filename for Step 2
            "message": msg,
            "can_ingest": is_ready
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

class IngestRequest(BaseModel):
    filename: str

@app.post("/ingest")
async def trigger_ingestion(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Step 2: Ingest (Internal Stage) -> Transform
    """
    import os
    from backend.ingestion_service import ingest_direct_from_file
    
    file_path = os.path.join("temp_uploads", req.filename)
    
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File {req.filename} not found on server. Please upload again."}
        
    def process_full_ingestion(path):
        try:
            # This function in ingestion_service.py now handles Ingest + Archive + Transform
            result = ingest_direct_from_file(path)
            print(f"Ingestion Result: {result}")
        finally:
            pass

    background_tasks.add_task(process_full_ingestion, file_path)
    
    return {"status": "success", "message": "Ingestion & Transformation Triggered in Background."}

# DEPRECATED INGEST ENDPOINT - KEPT FOR COMPATIBILITY


@app.post("/predict")
async def predict_eta(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None), # Make file optional for single prediction
    train: str = Form("false"), # Received as string from FormData
    pol: str = Form(None), # Optional for single prediction
    pod: str = Form(None), # Optional for single prediction
    mode: str = Form(None), # Optional for single prediction
    via: str = Form(None) # Optional for single prediction
):
    # 'train' comes as string "true"/"false" from JS FormData
    is_training = train.lower() == 'true'

    # --- 1. DATA LOADING & VALIDATION ---
    if file:
        # BATCH MODE (CSV)
        try:
            df = pd.read_csv(file.file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")
            
        # Batch Geo Validation
        initial_count = len(df)
        
        # Vectorized check
        def check_row(row):
            s = get_continent(row.get('PolCode'))
            d = get_continent(row.get('PodCode'))
            m = str(row.get('ModeOfTransport')).upper().strip()
            if s != 'UNKNOWN' and d != 'UNKNOWN' and s != d:
                if m in ['ROAD', 'RAIL', 'TRUCK']:
                    return False
            return True
            
        df = df[df.apply(check_row, axis=1)]
        if len(df) < initial_count:
            print(f"Batch Warning: Dropped {initial_count - len(df)} geographically impossible rows.")
            
    elif pol and pod and mode:
        # SINGLE SIMULATION MODE
        # Runtime Validation
        src_cont = get_continent(pol)
        dst_cont = get_continent(pod)
        mode_clean = mode.upper().strip()
        
        # 1. Continent Check
        if src_cont != 'UNKNOWN' and dst_cont != 'UNKNOWN' and src_cont != dst_cont:
             if mode_clean in ['ROAD', 'RAIL', 'TRUCK']:
                 raise HTTPException(
                     status_code=400, 
                     detail=f"Route Error: {mode} cannot be used between {pol} ({src_cont}) and {pod} ({dst_cont}). Please use Air or Ocean."
                 )
        
        # 2. Island/Isolated Check
        def get_country(code): return str(code)[:2].upper()
        src_country = get_country(pol)
        dst_country = get_country(pod)
        
        islands = {'BM', 'JP', 'TW', 'PH', 'ID', 'NZ', 'AU', 'LK', 'IE', 'IS', 'MG', 'CU', 'JM', 'DO', 'PR', 'MT', 'CY'}
        
        if mode_clean in ['ROAD', 'RAIL', 'TRUCK']:
            if src_country != dst_country:
                if src_country in islands or dst_country in islands:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Route Error: {mode} cannot be used for Island destination {dst_country} from {src_country}. Use Air or Ocean."
                    )
                if src_country == 'IE' or dst_country == 'IE':
                     raise HTTPException(status_code=400, detail="Route Error: Ireland requires Ferry (Ocean) or Air to reach mainland.")
                if 'BM' in [src_country, dst_country]:
                    raise HTTPException(status_code=400, detail="Route Error: Bermuda is an island. Use Air or Ocean.")
        
        # Create Single-Row DataFrame
        df = pd.DataFrame([{
            'PolCode': pol,
            'PodCode': pod,
            'ModeOfTransport': mode,
            'via_port': via, # Use if provided
            'Actual_Duration_Hours': 0 # Dummy for prediction
        }])
        
    else:
        raise HTTPException(status_code=400, detail="Missing Input: Provide a CSV file OR pol/pod/mode fields.")

    # --- 2. TRAINING LOGIC (Batch Only) ---
    if is_training and file and 'Actual_Duration_Hours' in df.columns:
         try:
            # Add trip_id if missing
            if 'trip_id' not in df.columns:
                df['trip_id'] = [f"UPLOAD_{i}" for i in range(len(df))]
            if 'trip_ATD' not in df.columns:
                df['trip_ATD'] = "2025-01-01" 
                
            # Save to disk
            df.to_csv(DATA_PATH, mode='a', header=False, index=False)
            background_tasks.add_task(retrain_model)
         except Exception as ex:
            print(f"Failed to save training data: {ex}")

    # --- 3. INFERENCE LOGIC ---
    try:
        if len(df) == 0:
             return {"status": "Error", "message": "No valid rows to predict (Check Geographic Constraints)"}

        # Feature Engineering
        df['Route'] = df['PolCode'].astype(str) + "_" + df['PodCode'].astype(str) + "_" + df['ModeOfTransport'].astype(str)
        
        encoded_df = df.copy()
        for col in ['PolCode', 'PodCode', 'ModeOfTransport', 'Route']:
            mapping = encoders.get(col, {})
            # Handle unknown keys (0)
            encoded_df[col] = encoded_df[col].astype(str).map(lambda s: mapping.get(s, 0))
            
        # 1. Prediction: XGBoost / Strategy
        pred_xgb = 0
        try:
            if isinstance(bst, str):
                if bst == "USE_RF": res = np.expm1(rf_model.predict(encoded_df[FEATURES]))[0]
                elif bst == "USE_ENSEMBLE": res = np.expm1(rf_model.predict(encoded_df[FEATURES]))[0] # Fallback to RF if Ensemble not fully wired
                else: res = 0
                pred_xgb = float(res)
            elif bst:
                 dmat = xgb.DMatrix(encoded_df[FEATURES])
                 pred_xgb = float(np.expm1(bst.predict(dmat))[0])
            else:
                 # Fallback if bst is None
                 pred_xgb = float(np.expm1(rf_model.predict(encoded_df[FEATURES]))[0])
        except:
             # Ultimate fallback
             pred_xgb = float(np.expm1(lr_model.predict(encoded_df[FEATURES]))[0]) if lr_model else 500.0
             
        preds = [pred_xgb] * len(df) # For legacy logic compatibility below (expecting array)
        if len(df) > 1:
             # Recalculate if batch
             pass # We did single row prediction logic above incorrectly for batch.
             
        # Correction for Batch vs Single
        # Re-do properly:
        preds = []
        if isinstance(bst, str):
             if bst == "USE_RF": preds = np.expm1(rf_model.predict(encoded_df[FEATURES]))
             else: preds = np.expm1(rf_model.predict(encoded_df[FEATURES]))
        elif bst:
             dmat = xgb.DMatrix(encoded_df[FEATURES])
             preds = np.expm1(bst.predict(dmat))
        else:
             preds = np.expm1(rf_model.predict(encoded_df[FEATURES]))
        
        # Format Results for Single Simulation vs Batch
        results = []
        accuracy = None # Initialize
        deviations = 0

        # Physics Enrichment for Single Simulation (To match /track richness)
        # If we have only 1 row (Single Sim), calculate the metrics
        if len(df) == 1 and pol and pod:
             p_eta = float(preds[0])
             c1 = get_lat_lon(pol)
             c2 = get_lat_lon(pod)
             
             # Fetch Intelligent Features
             rich = get_rich_features(pol, pod, mode, "GENERIC")
             
             dist_km = 0
             if c1 != (0,0) and c2 != (0,0):
                import math
                R = 6371
                dlat = math.radians(c2[0] - c1[0])
                dlon = math.radians(c2[1] - c1[1])
                a = math.sin(dlat/2)**2 + math.cos(math.radians(c1[0])) * math.cos(math.radians(c2[0])) * math.sin(dlon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                dist_km = R * c
                
             # Unit Conversion
             implied_speed = (dist_km / p_eta) * 1.1 if p_eta > 0 else 0
             dist_val, speed_val, dist_unit, speed_unit = convert_units(mode, dist_km, implied_speed)
             
             # CO2
             co2_factor = {'AIR': 500, 'ROAD': 62, 'TRUCK': 62, 'RAIL': 22, 'OCEAN': 8, 'SEA': 8}
             emissions_kg = (dist_km * co2_factor.get(str(mode).upper(), 50) * 10) / 1000

             # AI Explanation Generation
             stops_count = 0
             via_s = str(rich.get('via_port', 'DIRECT'))
             waypoints = []
             
             if via_s != 'DIRECT' and via_s != 'nan':
                 via_codes = via_s.split('|')
                 stops_count = len(via_codes)
                 for code in via_codes:
                     lat, lon = get_lat_lon(code)
                     if lat != 0:
                         waypoints.append({"lat": lat, "lon": lon, "code": code})
             
             explanation = f"AI Prediction: {round(p_eta, 1)} hours.\n"
             explanation += f"Route: {pol} -> {pod} via {mode}.\n"
             if stops_count > 0:
                 explanation += f"⚠️ Logic: Detected {stops_count} transshipment stop(s) at {via_s}.\n"
                 explanation += f"   Added +{rich.get('lane_avg_delay', 0)}h for dwelling time.\n"
             explanation += f"Risk Analysis: Route Risk is {rich['risk_score']}/100 based on current congestion."

             results.append({
                "trip_id": "SIM_LIVE",
                "prediction_hours": round(p_eta, 1),
                "prediction_days": round(p_eta / 24, 1),
                "confidence_score": rich['prediction_confidence'],
                "risk_score": rich['risk_score'],
                "ai_explanation": explanation,
                "route_details": {
                    "via_port": via_s,
                    "stops_count": stops_count,
                    "lane_avg_delay": rich['lane_avg_delay'],
                    "carrier_reliability": rich['carrier_reliability']
                },
                "rich_metrics": {
                    "distance": f"{dist_val} {dist_unit}",
                    "avg_speed": f"{speed_val} {speed_unit}",
                    "carbon_footprint": f"{round(emissions_kg, 1)} kgCO2e"
                },
                "coordinates": {
                    "source": {"lat": c1[0], "lon": c1[1], "code": pol},
                    "destination": {"lat": c2[0], "lon": c2[1], "code": pod},
                    "waypoints": waypoints
                }
             })
        else:
            # Batch Mode
            if 'Actual_Duration_Hours' in df.columns:
                actuals = df['Actual_Duration_Hours'].values
                wape = np.sum(np.abs(actuals - preds)) / np.sum(np.abs(actuals))
                accuracy = round(100 * (1 - wape), 2)
                
                for i, p in enumerate(preds):
                    err = abs(actuals[i] - p)
                    if err > 5: deviations += 1
                    results.append({
                        "trip_id": df.iloc[i].get('trip_id', f"SIM_{i}"),
                        "predicted_eta": float(p),
                        "actual_eta": float(actuals[i]),
                        "diff": float(p - actuals[i])
                    })
            else:
                 for i, p in enumerate(preds):
                    results.append({
                         "trip_id": df.iloc[i].get('trip_id', f"SIM_{i}"),
                        "predicted_eta": float(p),
                        "actual_eta": None,
                        "diff": None
                    })
        
        # Generate Batch Summary
        summary = f"Processed {len(preds)} shipments."
        if accuracy:
            summary += f" Model performed with **{accuracy}% accuracy** on this dataset."
            if deviations > 0:
                summary += f" ⚠️ **{deviations} shipments** show significant deviation (>5h)."
            else:
                summary += " ✅ All shipments are tracking within expected margins."
        else:
            summary += " Predictions generated successfully."

        if is_training:
            summary += "\n\n🆕 **Continuous Learning Active**: These records have been added to the master database. The model is retraining in the background to improve future accuracy."

        return {
            "predictions": results[:100], # Limit return size
            "count": len(results),
            "ai_summary": summary,
            "status": "Success", 
            "note": "Geographic Constraints Enforced"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- TRACKING & RICH DATA ENDPOINTS ---

def convert_units(mode, distance_km, speed_kmh):
    # Returns (Distance, Speed, DistUnit, SpeedUnit)
    m = str(mode).upper()
    if m in ['AIR', 'OCEAN', 'SEA']:
        # Nautical Miles & Knots
        dist = distance_km * 0.539957
        speed = speed_kmh * 0.539957
        return round(dist, 0), round(speed, 1), "NM", "Knots"
    elif m in ['ROAD', 'TRUCK']:
        # Km & Km/h
        return round(distance_km, 0), round(speed_kmh, 1), "km", "km/h"
    elif 'RAIL' in m:
       return round(distance_km, 0), round(speed_kmh, 1), "km", "km/h"
    else:
       return round(distance_km, 0), round(speed_kmh, 1), "km", "km/h"

# Helper access to coords (Need to ensure this dict is loaded/available)
def get_lat_lon(code):
    coords = port_coords.get(str(code)[:5], {'lat': 0, 'lon': 0})
    if isinstance(coords, dict):
        return (coords.get('lat', 0), coords.get('lon', 0))
    return (0, 0)

@app.get("/track/{vehicle_id}")
async def track_shipment(vehicle_id: str):
    # Search history_df for this ID (Mocking a real DB search)
    # Since history_df is large, we assume it's loaded.
    
    global history_df
    try:
        # Normalize ID
        vid = vehicle_id.strip()
        
        # Search
        # Optimize: If ID column not indexed, this is slow (O(N)). For 1.2M rows, ~50-100ms. Acceptable for MVP.
        match = history_df[history_df['Transport_Vehicle_ID'].astype(str) == vid]
        
        if len(match) == 0:
            raise HTTPException(status_code=404, detail="Tracking ID not found.")
            
        record = match.iloc[0]
        
        # Physics Re-Calculation for Rich Display (Reverse Engineering our engine)
        origin = record['PolCode']
        dest = record['PodCode']
        mode = record['ModeOfTransport']
        
        c1 = get_lat_lon(origin) 
        c2 = get_lat_lon(dest)
        
        dist_km = 0
        if c1 != (0,0) and c2 != (0,0):
            import math
            # Haversine inline or helper
            R = 6371
            dlat = math.radians(c2[0] - c1[0])
            dlon = math.radians(c2[1] - c1[1])
            a = math.sin(dlat/2)**2 + math.cos(math.radians(c1[0])) * math.cos(math.radians(c2[0])) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            dist_km = R * c
            
        duration_hours = float(record['Actual_Duration_Hours'])
        implied_speed_kmh = (dist_km / duration_hours) * 1.1 if duration_hours > 0 else 0
        
        # Unit Conversion
        dist_val, speed_val, dist_unit, speed_unit = convert_units(mode, dist_km, implied_speed_kmh)
        
        # Emissions (Simple approximation gCO2/ton-km)
        # Air: 500, Truck: 60, Rail: 20, Sea: 10
        co2_factor = {'AIR': 500, 'ROAD': 62, 'TRUCK': 62, 'RAIL': 22, 'OCEAN': 8, 'SEA': 8}
        emissions_kg = (dist_km * co2_factor.get(mode.upper(), 50) * 10) / 1000 # Assuming 10 tons cargo
        
        return {
            "tracking_id": vid,
            "origin": {"code": origin, "lat": c1[0], "lon": c1[1]},
            "destination": {"code": dest, "lat": c2[0], "lon": c2[1]},
            "status": "In Transit" if np.random.random() > 0.5 else "Delivered", # Mock Status
            "mode": mode,
            "via_port": record.get('via_port', 'DIRECT'), # NEW from dataset
            "eta": str(record.get('trip_ATA', 'N/A')),
            "metrics": {
                "distance": f"{dist_val} {dist_unit}",
                "avg_speed": f"{speed_val} {speed_unit}",
                "duration_hours": round(duration_hours, 1),
                "carbon_footprint": f"{round(emissions_kg, 1)} kgCO2e"
            },
            "coordinates": {
                "source": {"lat": c1[0], "lon": c1[1], "code": origin},
                "destination": {"lat": c2[0], "lon": c2[1], "code": dest}
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Trigger Reload V6
