import requests
import os

url = "http://127.0.0.1:8000/predict"
file_path = r"C:\Users\Administrator\.gemini\antigravity\PETA2\good_data_upload.csv"

# Check if requests is installed, if not try to install or use urllib (but for now let's assume valid env or use a simple try/except)
try:
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'train': 'true', 'model_type': 'xgb'}
        print(f"Uploading {file_path} to {url}...")
        response = requests.post(url, files=files, data=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
except NameError:
    print("Requests library not found, please install it or usage generic curl")
except Exception as e:
    print(f"Error: {e}")
