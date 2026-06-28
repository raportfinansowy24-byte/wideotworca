
import requests
import uuid
import os
import time

WORKER_API_KEY = os.getenv("WORKER_API_KEY")
API_URL = "http://localhost:5000/render-sequence"

data = {
    "topic": "3 niesamowite ciekawostki o kosmosie",
    "webhookUrl": "https://example.com/webhook"
}

headers = {
    "X-API-Key": WORKER_API_KEY,
    "Content-Type": "application/json",
    "Idempotency-Key": str(uuid.uuid4())
}

print(f"Sending request to {API_URL}...")
response = requests.post(API_URL, json=data, headers=headers)

if response.status_code == 202:
    job_id = response.json().get("job_id")
    print(f"Request accepted. Job ID: {job_id}")
    
    # Poll for completion
    while True:
        status_response = requests.get(f"http://localhost:5000/status/{job_id}", headers={"X-API-Key": WORKER_API_KEY})
        status_data = status_response.json()
        status = status_data.get("status")
        print(f"Status: {status}")
        
        if status == "success":
            print(f"Job completed! Video URL: {status_data.get('video_url')}")
            break
        elif status == "failed":
            print(f"Job failed: {status_data.get('error')}")
            break
        time.sleep(10)
else:
    print(f"Request failed: {response.status_code} - {response.text}")
