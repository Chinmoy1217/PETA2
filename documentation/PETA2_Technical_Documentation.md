# PETA2 - Predictive ETA System: Technical Documentation

## Project Overview

**PETA2 (Predictive ETA Analysis 2)** is an advanced shipment tracking and ETA prediction system designed to provide accurate delivery predictions for maritime and multimodal logistics. The system leverages machine learning, real-time data integration, and comprehensive risk analysis to deliver actionable insights to supply chain stakeholders.

---

## Architecture Overview

### Technology Stack

**Backend:**
- **Framework**: FastAPI (Python 3.10+)
- **ML Framework**: XGBoost (v3.1.3+), scikit-learn
- **Database**: Snowflake Data Warehouse (Cloud)
- **Data Processing**: Pandas, NumPy
- **Storage**: Azure Blob Storage (for file archival)

**Frontend:**
- React.js (not covered in this documentation)

**Infrastructure:**
- Containerized deployment (Docker/Docker Compose)
- RESTful API architecture
- JWT-based authentication

---

## Database Architecture (Snowflake)

### Schema Structure

The system uses a **three-layer data architecture**:

#### 1. **DT_PREP** (Staging/Preparation Layer)
- **PETA** table: Temporary staging for incoming shipment data
- Used for initial data loading and validation
- Truncated after successful transformation

#### 2. **DT_INGESTION** (Core Data Layer)

**Fact Tables:**
- **FACT_TRIP**: Core shipment records
  - `TRIP_ID`: Unique identifier
  - `POL` (Port of Loading), `POL_ATD`, `POL_ETD`
  - `POD` (Port of Discharge), `POD_ATD`, `POD_ETD`
  - `CARRIER_SCAC_CODE`, `VESSEL_IMO`
  
- **FACT_LANE_ETA_FEATURES**: Risk and feature metrics per lane
  - `LANE_NAME`: Route identifier (e.g., "Asia-North America")
  - `CARRIER_SCAC_CODE`
  - `SNAPSHOT_DATE`
  - **Risk Scores**:
    - `TOTAL_LANE_RISK_SCORE`
    - `WEATHER_RISK_SCORE`
    - `GEOPOLITICAL_RISK_SCORE`
    - `LABOR_STRIKE_SCORE`
    - `CUSTOMS_DELAY_SCORE`
    - `PORT_CONGESTION_SCORE`
    - `CARRIER_DELAY_SCORE`
    - `PEAK_SEASON_SCORE`
  - `BASE_ETA_DAYS`, `ADJUSTED_ETA_DAYS`

**Dimension Tables:**
- **DIM_CARRIER**: Carrier information
  - `CRR_SCAC_CD` (Primary Key)
  - `CRR_NAME`, `CRR_TYPE`
  - `IS_ACTIVE`, `SERVICE_START_DATE`

- **DIM_VEHICLE**: Vessel/vehicle details
  - `VEHICLE_NUMBER` (IMO number)
  - `VEHICLE_NM`, `VEHICLE_SECONDARY_NUMBER`, `VEHICLE_CALL_SIGN`

- **DIM_PORT**: Port master data
  - `CODE` (Port code - Primary Key)
  - `SOURCE_NAME`

#### 3. **Result/Prediction Tables**
- **PETA_RESULTS**: Prediction outputs
- **PETA_PREDICT_DLQ**: Dead Letter Queue for failed predictions

### Data Flow

```
Azure Blob (CSV) → DT_PREP.PETA → Transformation → DT_INGESTION (Facts/Dims)
                                       ↓
                            ML Model Training/Prediction
                                       ↓
                                 PETA_RESULTS
```

---

## Backend Components (Deep Dive)

### 1. **Data Ingestion Service** (`ingestion_service.py`)

**Purpose**: Orchestrates the ETL pipeline from Azure Blob Storage to Snowflake.

**Key Functions:**

#### `get_files_from_azure()`
- Connects to Azure Blob Storage via REST API
- Lists files in the `input` container
- Returns metadata for pending files

#### `snowflake_connection()`
- Establishes native Python connector to Snowflake
- Uses JWT token authentication
- Configuration from `config.py`

#### `copy_into_snowflake(conn, table_name, blob)`
- Uses Snowflake's `COPY INTO` command
- Loads CSV from Azure external stage
- Parameters:
  - `FILE_FORMAT`: CSV with headers, field delimiter `,`
  - `ON_ERROR`: Skip file on error
  - `FORCE`: True for reprocessing

#### `move_blob_to_archive(blob_name)`
- Moves successfully processed files to `archive` container
- REST API calls to Azure with SAS token

#### `ingest_direct_from_file(file_path)`
- Alternative ingestion bypassing Azure
- Direct file upload to Snowflake internal stage
- Uses `PUT` command for local files

**Workflow:**
```
1. Scan Azure → 2. COPY INTO PETA → 3. Archive File → 4. Trigger Transformation
```

---

### 2. **Data Transformation** (`data_transformation.py`)

**Purpose**: Transforms staging data into normalized fact/dimension tables.

**Key Function:** `run_transformation()`

**Transformation Logic:**

#### **FACT_TRIP Loading**
```sql
INSERT INTO FACT_TRIP (POL, POL_ATD, POL_ETD, POD, POD_ATD, POD_ETD, CARRIER_SCAC_CODE, VESSEL_IMO)
SELECT 
    PORT_OF_DEP,
    PORT_OF_DEP_ACT_DT,
    PORT_OF_DEP_EST_DT,
    PORT_OF_ARR,
    PORT_OF_ARRVL_ACT_DT,
    PORT_OF_ARRVL_EST_DT,
    CARRIER_SCAC_CODE,
    TRY_TO_NUMBER(VESSEL_IMO)
FROM DT_PREP.PETA
WHERE PORT_OF_DEP IS NOT NULL 
  AND PORT_OF_ARR IS NOT NULL
```

**Key Points:**
- Direct Snowflake-to-Snowflake INSERT (no Python loops)
- `TRY_TO_NUMBER()` handles non-numeric IMO values gracefully
- Logical mapping: POL = Departure, POD = Arrival

#### **DIM_CARRIER Loading**
- SQL-based UPSERT
- Deduplicates from both `CARRIER_SCAC_CODE` and `MASTER_CARRIER_SCAC_CODE`
- Sets default values: `CRR_TYPE='Ocean'`, `IS_ACTIVE=1`

#### **DIM_VEHICLE Loading**
- Filters invalid IMO numbers using `TRY_TO_NUMBER()`
- Prevents "Numeric value 'UNKNOWN' is not recognized" errors

#### **DIM_PORT Loading**
- Combines POL and POD codes
- Inserts only new ports (NOT EXISTS check)

**Post-Transformation:**
- Truncates `DT_PREP.PETA` to clear staging
- Commits all changes in single transaction

---

### 3. **Data Quality Check** (`quality_check.py`)

**Purpose**: Validates data before ingestion/transformation.

**Class:** `DataQualityChecker`

**Critical Fields (Accuracy Determinants):**
1. `port_of_arr` (Port of Arrival)
2. `port_of_dep` (Port of Departure)
3. `carrier_scac_code`
4. `master_carrier_scac_code`
5. `vessel_imo`

**Validation Rules:**
- **Completeness**: % of non-null values
- **Format Validation**:
  - Port codes: 5-char alphanumeric (e.g., `USLAX`)
  - SCAC codes: 4-char alphabetic (e.g., `MSCU`)
  - IMO numbers: 7-digit numeric or "UNKNOWN"
- **Accuracy Threshold**: Must achieve ≥80% accuracy across critical fields

**Output:**
```python
{
    "passed": True/False,
    "reason": "Explanation if failed",
    "metrics": {
        "total_rows": int,
        "port_of_arr_accuracy": float,
        "port_of_dep_accuracy": float,
        ...
    }
}
```

---

### 4. **Data Loader** (`data_loader.py`)

**Purpose**: Abstracts Snowflake queries for ML model training and dashboard.

**Key Methods:**

#### `get_training_view()`
Returns joined dataset for ML training:
```sql
WITH latest_lane_features AS (
    SELECT 
        LANE_NAME, 
        TOTAL_LANE_RISK_SCORE,
        BASE_ETA_DAYS,
        ... (all risk scores)
    FROM DT_INGESTION.FACT_LANE_ETA_FEATURES
    QUALIFY ROW_NUMBER() OVER (PARTITION BY LANE_NAME ORDER BY SNAPSHOT_DATE DESC) = 1
)
SELECT 
    t.TRIP_ID,
    t.POL as "PolCode",
    t.POD as "PodCode",
    'OCEAN' as "ModeOfTransport",
    t.CARRIER_SCAC_CODE as "Carrier",
    ABS(DATEDIFF('hour', t.POL_ATD, t.POD_ETD)) as "Actual_Duration_Hours",
    f.TOTAL_LANE_RISK_SCORE as "External_Risk_Score",
    f.BASE_ETA_DAYS,
    ... (all risk scores)
FROM DT_INGESTION.FACT_TRIP t
LEFT JOIN latest_lane_features f
  ON (lane_mapping_logic)
WHERE t.POL_ATD IS NOT NULL AND t.POD_ETD IS NOT NULL
```

**Lane Mapping Logic:**
- Converts port codes to regions:
  - `US%` → "North America"
  - `CN%`, `SG%`, `JP%`, `IN%` → "Asia"
  - `NL%`, `DE%`, `GB%` → "Europe"
- Joins on `Region_Origin-Region_Destination` = `LANE_NAME`

---

## Machine Learning Models (In-Depth)

### 1. **ETA Prediction Model** (`model_engine.py`)

**Algorithm**: XGBoost Gradient Boosting (Regression)

**Model Class:** `ETAModel`

#### **Features:**
1. **PolCode** (Port of Loading) - Categorical, Label Encoded
2. **PodCode** (Port of Discharge) - Categorical, Label Encoded
3. **ModeOfTransport** - Categorical, Label Encoded
4. **Severity_Score** - Numeric (from FACT_LANE_ETA_FEATURES)

#### **Target Variable:**
- `Actual_Duration_Hours / 24` (converted to days)

#### **Training Pipeline:**

**Step 1: Data Acquisition**
```python
def update_stats_from_csv(self, csv_path=None):
    loader = DataLoader()
    df = loader.get_training_view()  # Fetch from Snowflake
```

**Step 2: Data Quality Check**
```python
dq = QualityCheck.run_checks(df)
if not dq['passed']:
    return {'status': 'failed', 'reason': dq['reason']}
```

**Step 3: Feature Engineering**
- Calculate route statistics (mean, std per POL|POD|Mode)
- Cache risk factors per lane
- Fill missing `Severity_Score` with 0

**Step 4: Encoding**
```python
for col in ['PolCode', 'PodCode', 'ModeOfTransport']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    self.encoders[col] = le
```

**Step 5: Model Training**
```python
dtrain = xgb.DMatrix(X, label=y)
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05  # Learning rate
}
self.model = xgb.train(params, dtrain, num_boost_round=200)
```

**Step 6: Model Persistence**
- Saves to `eta_xgboost.json` (XGBoost format)
- Encoders to `label_encoders.joblib`
- Risk cache to `risk_cache.json`

#### **Prediction Logic:**

```python
def predict(self, pol, pod, mode):
    # 1. Lookup risk for route
    route_key = f"{pol}|{pod}|{mode}"
    risk_info = self.risk_cache.get(route_key, {'Severity_Score': 0})
    
    # 2. Encode inputs
    input_data = pd.DataFrame([{
        'PolCode': pol,
        'PodCode': pod,
        'ModeOfTransport': mode,
        'Severity_Score': severity
    }])
    # ... apply label encoders
    
    # 3. Predict
    dtest = xgb.DMatrix(input_data)
    pred_days = self.model.predict(dtest)[0]
    
    # 4. Generate explanation
    eta_date = (datetime.now() + timedelta(days=pred_days)).strftime("%Y-%m-%d")
    
    if severity \u003e 5:
        explanation = f"Significant delay due to {factors} (Risk: {severity}/100)"
    
    return {
        "predicted_days": pred_days,
        "eta_date": eta_date,
        "explanation": explanation
    }
```

**Explainability:**
- Risk-based explanations
- Maps severity scores to delay reasons (weather, geopolitical, strikes, etc.)

---

### 2. **Risk Classification Model** (`risk_model.py`)

**Algorithm**: Random Forest Classifier

**Purpose**: Predicts probability of shipment delay.

**Class:** `RiskModel`

#### **Features:**
- `External_Risk_Score`
- `WEATHER_RISK_SCORE`
- `PEAK_SEASON_SCORE`
- `PORT_CONGESTION_SCORE`
- `LABOR_STRIKE_SCORE`

#### **Target:**
- `Is_Delayed` (Binary)
  - 1 if `Actual_Duration_Hours \u003e median * 1.2`
  - 0 otherwise

#### **Training:**
```python
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X, y)
```

#### **Prediction:**
```python
def predict_risk_proba(self, features_dict):
    prob = self.model.predict_proba(input_data)[0][1]  # Class 1 probability
    return round(prob, 2)
```

**Output**: Float between 0.0 (low risk) and 1.0 (high risk)

---

### 3. **Carrier Reliability Model** (`reliability_model.py`)

**Type**: Statistical Aggregation (not ML)

**Purpose**: Scores carriers based on historical performance.

**Metrics:**
- **On-Time %**: Trips with `Actual_Duration \u003c= median * 1.1`
- **Average Delay**: Mean difference from median
- **Total Trips**: Sample size

**Scoring Formula:**
```python
score = on_time_pct * 100  # 0-100 scale
```

**Tiers:**
- **High Reliability**: Score \u003e 80
- **Medium Reliability**: 50 \u003c Score ≤ 80
- **Low Reliability**: Score ≤ 50

**Output:**
```json
{
    "MAERSK": {
        "score": 85.3,
        "avg_delay_hours": 12.5,
        "total_trips": 1523,
        "reliability_tier": "High"
    }
}
```

---

## Prediction Pipeline (`prediction_pipeline.py`)

**Purpose**: Automated batch prediction for new shipments.

### Workflow:

#### **Step 1: Fetch Incremental Data**
```python
def get_incremental_shipments(conn, limit=500):
    # Query: Get shipments newer than latest prediction
    SELECT * FROM DT_PREP.PETA
    WHERE ATD \u003e (SELECT MAX(PREDICTION_DATE) FROM PETA_RESULTS)
    LIMIT {limit}
```

#### **Step 2: Enrich with Risk Factors**
```python
def enrich_shipment(conn, carrier, pol, pod, dep_date):
    # Fetch from FACT_LANE_ETA_FEATURES
    # Match on CARRIER_SCAC_CODE + LANE_NAME + closest SNAPSHOT_DATE
```

#### **Step 3: Call Prediction API**
- HTTP POST to `http://127.0.0.1:8000/predict`
- Retry logic: 3 attempts with exponential backoff

#### **Step 4: Save Results**
```python
def save_prediction(conn, record):
    MERGE INTO PETA_RESULTS
    USING (SELECT ...) AS source
    ON (target.SHIPMENT_ID = source.SHIPMENT_ID)
    WHEN MATCHED THEN UPDATE ...
    WHEN NOT MATCHED THEN INSERT ...
```
- **Idempotent**: Prevents duplicate predictions

#### **Step 5: Error Handling**
```python
def insert_to_dlq(conn, payload, error_msg):
    INSERT INTO PETA_PREDICT_DLQ (DLQ_ID, PREDICTION_INPUT, ERROR_MSG)
    VALUES (UUID(), ?, ?)
```

---

## API Endpoints (FastAPI - `main.py` Overview)

### Authentication
- **POST `/login`**: JWT token generation
- **POST `/signup`**: User registration

### Core Endpoints

#### **POST `/simulate-trip`**
**Purpose**: Real-time ETA prediction

**Request:**
```json
{
    "PolCode": "CNSHA",
    "PodCode": "USLAX",
    "ModeOfTransport": "OCEAN",
    "Carrier": "MSCU",
    "trip_ATD": "2025-01-15",
    "congestion_level": 3,
    "weather_severity": 2
}
```

**Response:**
```json
{
    "predicted_days": 18.5,
    "eta_date": "2025-02-02",
    "confidence": 0.89,
    "explanation": "Minor impact from PORT_CONGESTION_SCORE",
    "route_stats": { "mean": 17.2, "std": 2.1 },
    "risk_score": 0.23
}
```

#### **POST `/upload-to-cloud`**
**Purpose**: File upload \u003e Quality Check \u003e Archive

**Workflow:**
1. Save file locally
2. Upload to Azure Blob (archive)
3. Run `DataQualityChecker`
4. Return validation results

#### **POST `/ingest`**
**Purpose**: Trigger ingestion pipeline

**Request:**
```json
{
    "filename": "shipment_data_20250115.csv"
}
```

**Background Tasks:**
- Calls `run_ingestion()` asynchronously
- Triggers `run_transformation()` on success

#### **GET `/active-shipments`**
**Purpose**: Dashboard data (top 50 shipments)

**Response:**
```json
[
    {
        "id": "TRIP-12345",
        "origin": "CNSHA",
        "destination": "USLAX",
        "mode": "OCEAN",
        "status": "In Transit",
        "eta": "145h"
    }
]
```

---

## Key Technical Innovations

### 1. **Incremental ETL Architecture**
- Event-driven pipeline (Azure → Snowflake → Transform)
- Watermark-based incremental loading
- Dead Letter Queue (DLQ) for error handling

### 2. **Risk-Aware Predictions**
- Multi-factor risk scoring (8 dimensions)
- Lane-level feature engineering
- Explainable AI with risk attribution

### 3. **SQL-Based Transformations**
- No Python loops for bulk operations
- Leverages Snowflake's compute power
- `TRY_TO_NUMBER()` for data type safety

### 4. **Quality-Gated Pipeline**
- Pre-ingestion validation
- 80% accuracy threshold enforcement
- Metrics-driven data acceptance

### 5. **Model Retraining Capability**
- Self-updating route statistics
- Incremental learning from production data
- Automated encoder management

---

## Performance Characteristics

### Scalability
- **Batch Prediction**: 500 shipments/run
- **API Response Time**: \u003c500ms (single prediction)
- **ETL Processing**: ~10,000 records in 2-3 minutes

### Accuracy
- **ETA Prediction**: RMSE ~0.52 days (reported)
- **Risk Model**: 95% classification accuracy (simulated)
- **Data Quality**: Enforced 80% minimum threshold

---

## File Structure Summary

```
PETA2/
├── backend/
│   ├── config.py                 # Snowflake credentials
│   ├── data_loader.py            # Snowflake query abstraction
│   ├── data_transformation.py    # ETL transformation logic
│   ├── ingestion_service.py      # Azure → Snowflake pipeline
│   ├── quality_check.py          # Data validation
│   ├── model_engine.py           # XGBoost ETA model
│   ├── risk_model.py            # Risk classification
│   ├── reliability_model.py      # Carrier scoring
│   ├── prediction_pipeline.py    # Batch prediction orchestrator
│   └── main.py                  # FastAPI application
├── model/
│   ├── eta_xgboost.json         # Trained model
│   ├── encoders.json            # Label encoders
│   ├── mode_stats.json          # Route statistics
│   └── port_coordinates.json    # Geospatial data
├── .env                         # Environment variables
└── ETA_Training_Walkthrough.ipynb  # ML training notebook
```

---

## Deployment Considerations

### Prerequisites
1. **Snowflake Account**: With DT_INGESTION schema
2. **Azure Blob Storage**: For file staging
3. **Python 3.10+**: With dependencies from `requirements.txt`

### Environment Variables
```bash
SNOWFLAKE_PASSWORD=\u003cJWT_TOKEN\u003e
AZURE_STORAGE_SAS_TOKEN=\u003cSAS_TOKEN\u003e
```

### Docker Deployment
```bash
docker-compose up -d
```

Exposes FastAPI on `http://localhost:8000`

---

## Security Features

1. **JWT Authentication**: All endpoints protected (except `/login`, `/signup`)
2. **SAS Token**: Azure Blob access with time-limited tokens
3. **Snowflake JWT**: Password-less authentication
4. **CORS Protection**: Configurable allowed origins

---

## Future Enhancements (Implicit from Code)

1. **Real-time Streaming**: Replace batch with Kafka/streaming ingestion
2. **Multi-Model Ensemble**: Combine XGBoost with LSTM for temporal patterns
3. **AutoML Integration**: Automated hyperparameter tuning
4. **GraphQL API**: For flexible frontend queries
5. **Kubernetes Orchestration**: For production-grade scaling

---

## Conclusion

PETA2 represents a comprehensive, production-ready ETA prediction system combining:
- **Robust Data Engineering**: Snowflake-based data warehouse with quality gates
- **Advanced ML**: XGBoost regression with risk-based feature engineering
- **Scalable Architecture**: API-first design with async processing
- **Operational Excellence**: DLQ, monitoring, and incremental learning

The system achieves high accuracy through multi-source data integration, rigorous validation, and continuous model improvement workflows.
