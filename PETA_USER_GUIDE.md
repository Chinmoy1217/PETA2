# PETA Prediction System - User Guide

## Quick Start: Upload CSV and Get Predictions

Your PETA (Predicted ETA) model is **LIVE** on `localhost:8005` with **87% accuracy** trained on 118,652 shipments.

### Method 1: File Upload (Batch Predictions)

**Input CSV Format:**
```csv
PolCode,PodCode,ModeOfTransport,Carrier,trip_ATD
CNSHG,USLAX,OCEAN,MAEU,2024-01-15
INNSA,DEHAM,OCEAN,MSCU,2024-02-20
SGSIN,USNYC,OCEAN,COSCO,2024-03-10
```

**Required Columns:**
- `PolCode`: Origin Port Code (e.g., CNSHG for Shanghai)
- `PodCode`: Destination Port Code (e.g., USLAX for Los Angeles)
- `ModeOfTransport`: Transport mode (OCEAN, AIR, ROAD, RAIL)
- `Carrier`: Carrier SCAC code (e.g., MAEU for Maersk)
- `trip_ATD`: Departure date (YYYY-MM-DD format)

**Run:**
```bash
python test_batch_upload.py
```

---

### Method 2: Single Prediction API

**Endpoint:** `POST http://localhost:8005/simulate`

**JSON Body:**
```json
{
  "PolCode": "CNSHG",
  "PodCode": "US LAX",
  "ModeOfTransport": "OCEAN",
  "Carrier": "MAEU",
  "trip_ATD": "2024-06-15",
  "congestion_level": 0,
  "weather_severity": 0
}
```

---

### Sample Files Provided

1. **`sample_shipments.csv`** - Example input with 5 routes
2. **`test_batch_upload.py`** - Script to test batch upload
3. **`test_peta.py`** - Script to test single predictions

---

### Model Performance

- **Accuracy**: 87.0%
- **MAE**: 94.72 hours (~3.9 days)
- **Model**: Ensemble (XGBoost + Random Forest)
- **Training Data**: 118,652 historical shipments
- **Features**: 20 predictors including seasonality, route patterns, carrier performance, weather/geopolitical risks

---

### Example Output

```
Shipment 1:
  Route: CNSHG → USLAX
  PETA: 456.3 hours (19.0 days)
  Departure: 2024-01-15
  Arrival: 2024-02-03 08:18
```

---

### Troubleshooting

If you get errors:
1. Ensure server is running on port 8005
2. Check CSV column names match exactly (case-sensitive)
3. Verify date format is YYYY-MM-DD
4. Valid port codes are required (e.g., CNSHG, USLAX, DEHAM)
