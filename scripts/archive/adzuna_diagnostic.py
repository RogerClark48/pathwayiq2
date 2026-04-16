"""
adzuna_diagnostic.py — Diagnostic test of Adzuna salary history endpoint.

Run:
    C:\Dev\pathwayiq2\venv\Scripts\python.exe scripts/adzuna_diagnostic.py
"""

import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

APP_ID  = os.environ.get("ADZUNA_APP_ID", "")
APP_KEY = os.environ.get("ADZUNA_APP_KEY", "")

if not APP_ID or not APP_KEY:
    print("ERROR: ADZUNA_APP_ID or ADZUNA_APP_KEY not found in .env")
    raise SystemExit(1)

BASE_URL = "https://api.adzuna.com/v1/api/salary/gb/history"

# Curated sample: 1-2 per broad area + one deliberately obscure title
TEST_TITLES = [
    # Digital
    "software engineer",
    "data scientist",
    # Construction
    "civil engineer",
    "quantity surveyor",
    # Engineering / electrical
    "electrical engineer",
    "mechanical engineer",
    # Life sciences
    "biochemist",
    "pharmacologist",
    # Geo sciences
    "geophysicist",
    "water engineer",
    # Energy
    "renewable energy engineer",
    # Manufacturing
    "metallurgist",
    # Deliberately obscure
    "geotechnician",
]


def query_adzuna(title: str) -> tuple[int, dict]:
    params = {
        "app_id":     APP_ID,
        "app_key":    APP_KEY,
        "title_only": title,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=15)
        return resp.status_code, resp.json()
    except requests.RequestException as e:
        return 0, {"error": str(e)}
    except ValueError:
        return resp.status_code, {"error": "non-JSON response"}


print(f"Adzuna salary history — diagnostic run")
print(f"{'=' * 60}\n")

for title in TEST_TITLES:
    print(f"TITLE: {title}")
    status, data = query_adzuna(title)

    # One-line summary
    fields = list(data.keys()) if isinstance(data, dict) else []
    month_count = 0
    has_salary = False

    if isinstance(data, dict):
        # Response may be {"month": [...]} or {"1": {...}, "2": {...}} depending on version
        # Check for any list/dict values that look like time series data
        for v in data.values():
            if isinstance(v, list) and len(v) > 0:
                has_salary = True
                month_count = len(v)
            elif isinstance(v, dict) and "value" in v:
                has_salary = True
                month_count += 1

    print(f"  Status: {status}  |  Fields: {fields}  |  Salary data: {has_salary}  |  Data points: {month_count}")
    print(f"  Raw JSON:")
    print(json.dumps(data, indent=4))
    print()

    time.sleep(1)
