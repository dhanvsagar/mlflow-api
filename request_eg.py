import requests
import json

url = "http://127.0.0.1:1234/invocations"
headers = {"Content-Type": "application/json"}

payload = {
    "dataframe_split": {
        "columns": ["x"],
        "data": [[2.5], [7.0]]
    }
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())