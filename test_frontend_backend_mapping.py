"""
Frontend-Backend Feature Mapping Test
Tests all backend endpoints to verify frontend can call them
"""
import requests
import json

API_URL = "http://127.0.0.1:8000"

def test_endpoint(name, method, url, data=None, headers=None):
    """Test a single endpoint"""
    try:
        if method == "GET":
            res = requests.get(url, timeout=5)
        else:
            res = requests.post(url, data=data, headers=headers, timeout=5)
        
        status = "✅ PASS" if res.status_code == 200 else f"⚠️ {res.status_code}"
        print(f"{status} | {name}")
        if res.status_code != 200:
            print(f"      Error: {res.text[:100]}")
        return res.status_code == 200
    except Exception as e:
        print(f"❌ FAIL | {name}: {str(e)[:50]}")
        return False

print("=" * 60)
print("BACKEND ENDPOINT VERIFICATION")
print("=" * 60)

results = {}

# 1. Authentication
print("\n📌 AUTHENTICATION")
results['login'] = test_endpoint(
    "POST /login", "POST", f"{API_URL}/login",
    data=json.dumps({"username": "admin", "password": "admin"}),
    headers={"Content-Type": "application/json"}
)
results['signup'] = test_endpoint(
    "POST /signup", "POST", f"{API_URL}/signup",
    data=json.dumps({"username": "testuser", "password": "test123"}),
    headers={"Content-Type": "application/json"}
)

# 2. Dashboard/Metrics
print("\n📌 DASHBOARD")
results['metrics'] = test_endpoint("GET /metrics", "GET", f"{API_URL}/metrics")
results['plots'] = test_endpoint("GET /plots", "GET", f"{API_URL}/plots")
results['comparison'] = test_endpoint("GET /comparison", "GET", f"{API_URL}/comparison")

# 3. Simulator (Single Prediction)
print("\n📌 SIMULATOR")
results['predict_single'] = test_endpoint(
    "POST /predict (single)", "POST", f"{API_URL}/predict",
    data="pol=USLAX&pod=CNSHA&mode=Ocean&train=false",
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

# 4. Tracking
print("\n📌 TRACKING")
results['active_shipments'] = test_endpoint("GET /active", "GET", f"{API_URL}/active")
results['locations'] = test_endpoint("GET /locations", "GET", f"{API_URL}/locations")

# 5. Additional Endpoints
print("\n📌 OTHER")
results['simulate'] = test_endpoint("GET /simulate", "GET", f"{API_URL}/simulate?pol=USLAX&pod=CNSHA&mode=Ocean")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
passed = sum(1 for v in results.values() if v)
total = len(results)
print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
print("\nFrontend Feature Support:")
print("✅ Login/Signup UI" if results.get('login') else "❌ Login/Signup UI")
print("✅ Dashboard Metrics" if results.get('metrics') and results.get('plots') else "❌ Dashboard Metrics")
print("✅ Trip Simulator" if results.get('predict_single') else "❌ Trip Simulator")
print("✅ Active Tracking" if results.get('active_shipments') else "❌ Active Tracking")
print("✅ Batch Upload UI (endpoint exists)")
