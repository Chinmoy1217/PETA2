# /simulate Endpoint - Request Template

## COMPLETE WORKING FORMAT (JSON)

```json
{
  "PolCode": "CNSHG",
  "PodCode": "USLAX",
  "ModeOfTransport": "OCEAN",
  "Carrier": "MAEU",
  "trip_ATD": "2024-06-15",
  "congestion_level": 50,
  "weather_severity": 20
}
```

---

## Field Descriptions

| Field | Required | Format | Example | Description |
|-------|----------|--------|---------|-------------|
| `PolCode` | ✅ Yes | String (5 chars) | `"CNSHG"` | Origin port code (Country+Port) |
| `PodCode` | ✅ Yes | String (5 chars) | `"USLAX"` | Destination port code |
| `ModeOfTransport` | ✅ Yes | String | `"OCEAN"` | Transport mode (OCEAN/AIR/ROAD/RAIL) |
| `Carrier` | ✅ Yes | String (4 chars) | `"MAEU"` | Carrier SCAC code |
| `trip_ATD` | ✅ Yes | Date (YYYY-MM-DD) | `"2024-06-15"` | Departure date |
| `congestion_level` | ✅ Yes | Integer (0-100) | `50` | Port congestion severity |
| `weather_severity` | ✅ Yes | Integer (0-100) | `20` | Weather risk level |

**Congestion Level Scale:**
- `0-20`: Normal operations
- `21-50`: Moderate congestion (1-2 day delays)
- `51-80`: Heavy congestion (3-5 day delays)
- `81-100`: Severe congestion (5+ day delays)

**Weather Severity Scale:**
- `0-20`: Clear conditions
- `21-50`: Minor weather issues
- `51-80`: Significant storms/typhoons
- `81-100`: Extreme weather events

---

## Common Port Codes

**Asia:**
- `CNSHG` - Shanghai, China
- `SGSIN` - Singapore
- `HKHKG` - Hong Kong
- `KRPUS` - Busan, South Korea
- `INNSA` - Nhava Sheva, India

**North America:**
- `USLAX` - Los Angeles, USA
- `USNYC` - New York, USA
- `USLGB` - Long Beach, USA

**Europe:**
- `DEHAM` - Hamburg, Germany
- `NLRTM` - Rotterdam, Netherlands
- `BEANR` - Antwerp, Belgium

**Middle East:**
- `AEJEA` - Jebel Ali, UAE

---

## Common Carrier SCAC Codes

- `MAEU` - Maersk
- `MSCU` - MSC (Mediterranean Shipping Company)
- `COSCO` - COSCO
- `CMAU` - CMA CGM
- `EGLV` - Evergreen
- `HLCU` - Hapag-Lloyd
- `ONEY` - ONE (Ocean Network Express)
- `HMMU` - HMM

---

## Example Requests

### Python (requests)
```python
import requests

payload = {
    "PolCode": "CNSHG",
    "PodCode": "USLAX",
    "ModeOfTransport": "OCEAN",
    "Carrier": "MAEU",
    "trip_ATD": "2024-06-15",
    "congestion_level": 50,
    "weather_severity": 20
}

response = requests.post("http://localhost:8005/simulate", json=payload)
print(response.json())
```

### cURL
```bash
curl -X POST http://localhost:8005/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "PolCode": "CNSHG",
    "PodCode": "USLAX",
    "ModeOfTransport": "OCEAN",
    "Carrier": "MAEU",
    "trip_ATD": "2024-06-15"
  }'
```

### JavaScript (fetch)
```javascript
fetch('http://localhost:8005/simulate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    PolCode: "CNSHG",
    PodCode: "USLAX",
    ModeOfTransport: "OCEAN",
    Carrier: "MAEU",
    trip_ATD: "2024-06-15"
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## Response Format

```json
{
  "prediction_hours": 456.3,
  "base_eta": "18.5 days",
  "risk_factors": {
    "weather": 5,
    "congestion": 3,
    "geopolitical": 2
  },
  "model_candidates": [
    {"name": "XGBoost", "pred": 458.1},
    {"name": "Random Forest", "pred": 454.5}
  ],
  "ai_explanation": "Model consensus indicates...",
  "status": "Success"
}
```

---

## Quick Copy-Paste Templates

**Trans-Pacific (Asia → USA):**
```json
{"PolCode": "CNSHG", "PodCode": "USLAX", "ModeOfTransport": "OCEAN", "Carrier": "MAEU", "trip_ATD": "2024-06-15", "congestion_level": 50, "weather_severity": 20}
```

**Europe → USA:**
```json
{"PolCode": "DEHAM", "PodCode": "USNYC", "ModeOfTransport": "OCEAN", "Carrier": "HMMU", "trip_ATD": "2024-07-01", "congestion_level": 30, "weather_severity": 10}
```

**Asia → Europe:**
```json
{"PolCode": "SGSIN", "PodCode": "NLRTM", "ModeOfTransport": "OCEAN", "Carrier": "MSCU", "trip_ATD": "2024-08-10", "congestion_level": 40, "weather_severity": 25}
```

**Normal Conditions (Low Risk):**
```json
{"PolCode": "CNSHG", "PodCode": "USLAX", "ModeOfTransport": "OCEAN", "Carrier": "MAEU", "trip_ATD": "2024-06-15", "congestion_level": 10, "weather_severity": 5}
```

**High Risk Scenario:**
```json
{"PolCode": "CNSHG", "PodCode": "USLAX", "ModeOfTransport": "OCEAN", "Carrier": "MAEU", "trip_ATD": "2024-06-15", "congestion_level": 85, "weather_severity": 70}
```
