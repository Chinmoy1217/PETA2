
import urllib.request
import urllib.parse
import json
import mimetypes
import uuid

url = "http://127.0.0.1:8000/batch-predict"
filepath = "prediction_input.csv"

# Prepare multipart file upload using standard library
boundary = uuid.uuid4().hex
headers = {
    'Content-Type': f'multipart/form-data; boundary={boundary}'
}

with open(filepath, 'rb') as f:
    file_content = f.read()

# Build body
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="{filepath}"\r\n'
    f'Content-Type: text/csv\r\n\r\n'
).encode('utf-8') + file_content + f'\r\n--{boundary}--\r\n'.encode('utf-8')

req = urllib.request.Request(url, data=body, headers=headers, method='POST')

print(f"Fetching predictions for {filepath}...")
try:
    with urllib.request.urlopen(req) as response:
        data = json.load(response)
        
        if data.get("status") == "success":
            print(f"\nSUCCESS! Uploaded to Table: {data.get('snowflake_table')}")
            print(f"Processed {len(data.get('peta_predictions', []))} trips.\n")
            print("--- RESULTS (First 10 shown, full list saved) ---")
            
            preds = data.get('peta_predictions', [])
            for i, p in enumerate(preds):
                print(f"Trip {p['trip_id']}: {p['PETA_Predicted_Duration']} hours")
                if i >= 50: break # Print all 50
        else:
            print("Server returned error:")
            print(json.dumps(data, indent=2))
            
except Exception as e:
    print(f"Failed to fetch results: {e}")
