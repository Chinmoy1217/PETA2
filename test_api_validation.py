"""
PETA API Complete Parameter Validation Test
Tests all parameters against Snowflake table schema
"""
import requests
import json
from datetime import datetime

API_URL = "http://localhost:8005/simulate"

# ============================================================================
# REQUIRED API PARAMETERS (All 7 fields)
# ============================================================================

REQUIRED_PARAMETERS = {
    "PolCode": {
        "type": "string",
        "length": "5 chars",
        "example": "CNSHG",
        "description": "Origin Port Code (Country + Port)",
        "snowflake_field": "POL_CODE"
    },
    "PodCode": {
        "type": "string", 
        "length": "5 chars",
        "example": "USLAX",
        "description": "Destination Port Code",
        "snowflake_field": "POD_CODE"
    },
    "ModeOfTransport": {
        "type": "string",
        "allowed_values": ["OCEAN", "AIR", "ROAD", "RAIL"],
        "example": "OCEAN",
        "description": "Transport mode",
        "snowflake_field": "MODE_OF_TRANSPORT"
    },
    "Carrier": {
        "type": "string",
        "length": "4 chars",
        "example": "MAEU",
        "description": "Carrier SCAC code",
        "snowflake_field": "CARRIER_SCAC_CODE"
    },
    "trip_ATD": {
        "type": "date",
        "format": "YYYY-MM-DD",
        "example": "2024-06-15",
        "description": "Departure date",
        "snowflake_field": "DEPARTURE_DATE"
    },
    "congestion_level": {
        "type": "integer",
        "range": "0-100",
        "example": 50,
        "description": "Port congestion severity",
        "snowflake_field": "CONGESTION_LEVEL"
    },
    "weather_severity": {
        "type": "integer",
        "range": "0-100",
        "example": 20,
        "description": "Weather risk level",
        "snowflake_field": "WEATHER_SEVERITY"
    }
}

print("=" * 100)
print("  PETA API PARAMETER VALIDATION TEST")
print("=" * 100)
print()

# Display Parameter Requirements
print("REQUIRED API PARAMETERS (7 fields):")
print("-" * 100)
for param, details in REQUIRED_PARAMETERS.items():
    extra_info = details.get('length', '') or details.get('range', '') or details.get('format', '')
    print(f"{param:25} | {details['type']:8} | {extra_info:15} | → {details['snowflake_field']}")
print()

# ============================================================================
# TEST CASES
# ============================================================================

test_cases = [
    {
        "name": "✅ VALID - Complete Request",
        "data": {
            "PolCode": "CNSHG",
            "PodCode": "USLAX",
            "ModeOfTransport": "OCEAN",
            "Carrier": "MAEU",
            "trip_ATD": "2024-06-15",
            "congestion_level": 50,
            "weather_severity": 20
        },
        "should_pass": True
    },
    {
        "name": "✅ VALID - Low Risk",
        "data": {
            "PolCode": "SGSIN",
            "PodCode": "DEHAM",
            "ModeOfTransport": "OCEAN",
            "Carrier": "MSCU",
            "trip_ATD": "2024-07-01",
            "congestion_level": 10,
            "weather_severity": 5
        },
        "should_pass": True
    },
    {
        "name": "✅ VALID - High Risk",
        "data": {
            "PolCode": "INNSA",
            "PodCode": "USNYC",
            "ModeOfTransport": "OCEAN",
            "Carrier": "COSCO",
            "trip_ATD": "2024-08-10",
            "congestion_level": 85,
            "weather_severity": 70
        },
        "should_pass": True
    },
    {
        "name": "❌ INVALID - Missing congestion_level",
        "data": {
            "PolCode": "CNSHG",
            "PodCode": "USLAX",
            "ModeOfTransport": "OCEAN",
            "Carrier": "MAEU",
            "trip_ATD": "2024-06-15",
            "weather_severity": 20
        },
        "should_pass": False
    },
    {
        "name": "❌ INVALID - Missing weather_severity",
        "data": {
            "PolCode": "CNSHG",
            "PodCode": "USLAX",
            "ModeOfTransport": "OCEAN",
            "Carrier": "MAEU",
            "trip_ATD": "2024-06-15",
            "congestion_level": 50
        },
        "should_pass": False
    },
    {
        "name": "❌ INVALID - Wrong date format",
        "data": {
            "PolCode": "CNSHG",
            "PodCode": "USLAX",
            "ModeOfTransport": "OCEAN",
            "Carrier": "MAEU",
            "trip_ATD": "06/15/2024",  # Wrong format
            "congestion_level": 50,
            "weather_severity": 20
        },
        "should_pass": False
    }
]

# Run Tests
print("=" * 100)
print("RUNNING PARAMETER VALIDATION TESTS")
print("=" * 100)
print()

passed = 0
failed = 0

for i, test in enumerate(test_cases, 1):
    print(f"Test {i}: {test['name']}")
    print("-" * 100)
    
    # Show request
    print(f"Request: {json.dumps(test['data'], indent=2)}")
    
    try:
        response = requests.post(API_URL, json=test['data'], timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            pred_hours = data.get('prediction_hours')
            
            if pred_hours:
                print(f"✅ API Response: SUCCESS")
                print(f"   Predicted Hours: {pred_hours:.1f}")
                print(f"   Will map to Snowflake:")
                for api_param, details in REQUIRED_PARAMETERS.items():
                    value = test['data'].get(api_param)
                    sf_field = details['snowflake_field']
                    print(f"     {api_param} → {sf_field} = {value}")
                
                if test['should_pass']:
                    print(f"   ✅ PASS - Expected to succeed")
                    passed += 1
                else:
                    print(f"   ⚠️  UNEXPECTED - Expected to fail but passed")
                    failed += 1
            else:
                print(f"❌ API Response: Missing prediction field")
                print(f"   Available: {list(data.keys())}")
                failed += 1
        else:
            print(f"❌ API Response: HTTP {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            
            if not test['should_pass']:
                print(f"   ✅ PASS - Expected to fail")
                passed += 1
            else:
                print(f"   ❌ FAIL - Expected to succeed")
                failed += 1
                
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}: {str(e)}")
        if not test['should_pass']:
            print(f"   ✅ PASS - Expected to fail")
            passed += 1
        else:
            print(f"   ❌ FAIL - Expected to succeed")
            failed += 1
    
    print()

# Summary
print("=" * 100)
print("TEST SUMMARY")
print("=" * 100)
print(f"Total Tests: {len(test_cases)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print()

# Snowflake Mapping Reference
print("=" * 100)
print("SNOWFLAKE TABLE MAPPING")
print("=" * 100)
print()
print("API Parameter          → Snowflake Column         → Stored As")
print("-" * 100)
mapping = [
    ("PolCode", "POL_CODE", "VARCHAR(10)"),
    ("PodCode", "POD_CODE", "VARCHAR(10)"),
    ("ModeOfTransport", "MODE_OF_TRANSPORT", "VARCHAR(20)"),
    ("Carrier", "CARRIER_SCAC_CODE", "VARCHAR(10)"),
    ("trip_ATD", "DEPARTURE_DATE", "DATE"),
    ("congestion_level", "CONGESTION_LEVEL", "NUMBER(3,0)"),
    ("weather_severity", "WEATHER_SEVERITY", "NUMBER(3,0)"),
    ("(calculated)", "PETA_HOURS", "NUMBER(10,2)"),
    ("(calculated)", "PETA_DAYS", "NUMBER(10,2)"),
    ("(calculated)", "PETA_DATE", "DATE"),
    ("(calculated)", "PETA_DATETIME", "TIMESTAMP_NTZ"),
    ("(future)", "ATA_DATE", "DATE - nullable"),
    ("(future)", "ATA_DATETIME", "TIMESTAMP_NTZ - nullable"),
    ("(future)", "VARIANCE_HOURS", "NUMBER(10,2) - nullable"),
    ("(system)", "STATUS", "'PREDICTED'"),
]

for api, snowflake, dtype in mapping:
    print(f"{api:25} → {snowflake:30} → {dtype}")

print()
print("=" * 100)
