# test_explain.py - Test DeepShield predict_with_explain endpoint

import requests
import os

# ---------------- CONFIG ----------------
# Use 127.0.0.1 for local testing
# Use LAN IP of the machine running backend for mobile testing
BASE_URL = "http://127.0.0.1:8000"  # Change to your backend IP if needed
ENDPOINT = "/predict_with_explain"
API_KEY = "mysecretkey"  # must match backend API_KEY

HEADERS = {
    "X-API-Key": API_KEY
}

# ---------------- IMAGE ----------------
IMAGE_PATH = "data/train/real/100_1.jpg"  # Make sure this file exists
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"{IMAGE_PATH} does not exist!")

# ---------------- REQUEST ----------------
with open(IMAGE_PATH, "rb") as f:
    files = {"file": (os.path.basename(IMAGE_PATH), f, "image/jpeg")}
    try:
        response = requests.post(f"{BASE_URL}{ENDPOINT}", files=files, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        print("✅ Response:", data)
    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh, response.text)
    except requests.exceptions.ConnectionError as errc:
        print("Connection Error:", errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        print("Request Exception:", err)
