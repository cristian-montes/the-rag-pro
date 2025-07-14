import requests

model_url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json"

try:
    response = requests.get(model_url, allow_redirects=True, timeout=10)
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    print(f"Successfully connected to: {model_url}")
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {response.headers}")
    print(f"Content (first 200 chars): {response.text[:200]}")
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e.response.status_code} - {e.response.reason}")
    print(f"Response text: {e.response.text}")
    print(f"Request URL: {e.request.url}")
    print(f"Request Headers: {e.request.headers}")
except requests.exceptions.ConnectionError as e:
    print(f"Connection Error: {e}")
    print("This often indicates a firewall, proxy, or DNS issue.")
except requests.exceptions.Timeout as e:
    print(f"Timeout Error: {e}")
    print("The request took too long to respond.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")