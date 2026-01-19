
import requests
import json

# Config
ACCOUNT = "COZENTUS-DATAPRACTICE"
USER = "HACKATHON_DT"
PASS = "Welcome@Start123"
WAREHOUSE = "COMPUTE_WH"
DATABASE = "HACAKATHON"
SCHEMA = "PUBLIC"

# URL Construction
# Account often needs region. If US-West-2, it's just account.
# If failed, try appending region.
BASE_URL = f"https://{ACCOUNT}.snowflakecomputing.com"
LOGIN_URL = f"{BASE_URL}/session/v1/login-request"

def test_login():
    print(f"Attempting login to: {LOGIN_URL}")
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Python/3.9",
        "Accept": "application/json"
    }
    
    body = {
        "data": {
            "ACCOUNT_NAME": ACCOUNT,
            "LOGIN_NAME": USER,
            "PASSWORD": PASS,
            "CLIENT_APP_ID": "PythonRestClient",
            "CLIENT_APP_VERSION": "1.0"
        }
    }
    
    try:
        # Verify=False to bypass ANY local SSL cert store issues (The 'Nuclear Option' for SSL errors)
        resp = requests.post(LOGIN_URL, json=body, headers=headers, verify=False)
        
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if data['success']:
                token = data['data']['token']
                print("SUCCESS: Got Session Token!")
                return token
            else:
                print(f"FAILED: {data['message']}")
        else:
            print(f"HTTP Error: {resp.text}")
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
        
    return None

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings() # Silence the SSL warning
    test_login()
