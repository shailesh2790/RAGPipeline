import requests
import json

url = "http://localhost:5000/query"
payload = {"query": "What is the capital of France?"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.json())