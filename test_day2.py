# test_day2.py
import requests

BASE = "http://127.0.0.1:5000"

# Test 1: Health check
r = requests.get(f"{BASE}/")
print("Health:", r.json())

# Test 2: Predict with fake landmarks
fake = [0.5] * 63
r = requests.post(f"{BASE}/predict", json={"landmarks": fake})
print("Predict:", r.json())

# Test 3: Predict 20 times to test sentence building
for _ in range(20):
    r = requests.post(f"{BASE}/predict", json={"landmarks": fake})
print("After 20 calls:", r.json())

# Test 4: Backspace
r = requests.post(f"{BASE}/sentence/backspace")
print("After backspace:", r.json())

# Test 5: Clear
r = requests.post(f"{BASE}/sentence/clear")
print("After clear:", r.json())