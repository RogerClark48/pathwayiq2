"""
adzuna_diagnostic2.py — Diagnostic test of three Adzuna jobs vertical endpoints:
  1. Histogram (salary distribution by keyword)
  2. History (historical average salary by category)
  3. Jobsworth (salary predictor by title + description)

Run:
    C:\Dev\pathwayiq2\venv\Scripts\python.exe scripts/adzuna_diagnostic2.py
"""

import json
import sqlite3
import time
import urllib.parse
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

AUTH = {"app_id": APP_ID, "app_key": APP_KEY}
BASE = "https://api.adzuna.com/v1/api/jobs/gb"
SEP  = "=" * 60


def get(url, params):
    try:
        r = requests.get(url, params=params, timeout=15)
        try:
            data = r.json()
        except ValueError:
            data = {"error": "non-JSON response", "body": r.text[:200]}
        return r.status_code, data
    except requests.RequestException as e:
        return 0, {"error": str(e)}


def print_response(status, data, label=""):
    summary_keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
    print(f"  Status: {status}  |  Top-level keys: {summary_keys}")
    if label:
        print(f"  ({label})")
    print(f"  Raw JSON:")
    print(json.dumps(data, indent=4))
    print()


# ---------------------------------------------------------------------------
# 1. Categories endpoint — get valid category tags first
# ---------------------------------------------------------------------------
print(SEP)
print("ENDPOINT 1a: CATEGORIES (to get valid category slugs)")
print(SEP)

status, data = get(f"{BASE}/categories", AUTH)
print_response(status, data)
time.sleep(1)

# ---------------------------------------------------------------------------
# 2. Histogram — salary distribution by keyword
# ---------------------------------------------------------------------------
print(SEP)
print("ENDPOINT 2: HISTOGRAM (salary distribution by keyword)")
print(SEP)

histogram_titles = [
    "software engineer",
    "civil engineer",
    "data scientist",
    "wind turbine technician",
]

first_histogram_shown = False
for title in histogram_titles:
    params = {**AUTH, "what": title}
    status, data = get(f"{BASE}/histogram", params)
    show_full = not first_histogram_shown and status == 200
    print(f"  TITLE: {title}")
    print(f"  Status: {status}  |  Keys: {list(data.keys()) if isinstance(data, dict) else '?'}")
    if show_full:
        print("  Full JSON (first successful response):")
        print(json.dumps(data, indent=4))
        first_histogram_shown = True
    elif isinstance(data, dict) and "exception" in data:
        print(f"  Exception: {data.get('exception')} — {data.get('display','')}")
    elif isinstance(data, dict):
        # Show a compact summary of what came back
        for k, v in data.items():
            if isinstance(v, dict):
                print(f"    {k}: {list(v.keys())[:10]}")
            else:
                print(f"    {k}: {str(v)[:80]}")
    print()
    time.sleep(1)

# ---------------------------------------------------------------------------
# 3. History — historical average salary by category
# ---------------------------------------------------------------------------
print(SEP)
print("ENDPOINT 3: HISTORY (historical average salary by category)")
print(SEP)

# Use actual category tags from the categories response if available, else guess
category_guesses = ["engineering-jobs", "it-jobs", "science-jobs"]

# Try to extract real tags from categories response above
try:
    real_tags = []
    cat_results = data  # last response — may not be categories; re-fetch
    status_c, data_c = get(f"{BASE}/categories", AUTH)
    if status_c == 200 and "results" in data_c:
        real_tags = [r.get("tag") for r in data_c["results"] if r.get("tag")]
        category_guesses = real_tags[:6]  # test first 6
        print(f"  Using real category tags from API: {category_guesses}\n")
except Exception:
    print(f"  Using guessed category tags: {category_guesses}\n")

time.sleep(1)
first_history_shown = False
for cat in category_guesses:
    params = {**AUTH, "category": cat}
    status, data = get(f"{BASE}/history", params)
    show_full = not first_history_shown and status == 200 and "month" in data
    print(f"  CATEGORY: {cat}")
    print(f"  Status: {status}  |  Keys: {list(data.keys()) if isinstance(data, dict) else '?'}")
    if show_full:
        # month data can be large — print first 3 entries only
        trimmed = {k: (dict(list(v.items())[:3]) if isinstance(v, dict) else v)
                   for k, v in data.items()}
        print("  Full JSON (trimmed to first 3 month entries):")
        print(json.dumps(trimmed, indent=4))
        first_history_shown = True
    elif isinstance(data, dict) and "exception" in data:
        print(f"  Exception: {data.get('exception')} — {data.get('display','')}")
    elif isinstance(data, dict) and "month" in data:
        months = data["month"]
        print(f"  month entries: {len(months)}")
        if months:
            print(f"  Sample entry: {json.dumps(list(months.items())[0], indent=4)}")
    print()
    time.sleep(1)

# ---------------------------------------------------------------------------
# 4. Jobsworth — salary predictor by title + description
# ---------------------------------------------------------------------------
print(SEP)
print("ENDPOINT 4: JOBSWORTH (salary predictor by title + description)")
print(SEP)

jobsworth_titles = [
    "software engineer",
    "civil engineer",
    "data scientist",
    "geotechnician",
    "wind turbine technician",
    "medical physicist",
]

conn = sqlite3.connect(DB_PATH := str(PROJECT_ROOT / "job_roles_asset.db"))
conn.row_factory = sqlite3.Row
cur = conn.cursor()

for title in jobsworth_titles:
    # Fetch description from DB — prefer NCS, fall back to any record
    cur.execute("""
        SELECT description, overview, title AS db_title
        FROM jobs
        WHERE normalized_title = ?
          AND (description IS NOT NULL OR overview IS NOT NULL)
        ORDER BY CASE source WHEN 'ncs' THEN 0 ELSE 1 END
        LIMIT 1
    """, (title,))
    row = cur.fetchone()

    if row:
        desc_text = (row["description"] or row["overview"] or "")[:1000]
        db_title  = row["db_title"]
    else:
        desc_text = f"A professional working as a {title}."
        db_title  = title
        print(f"  (no DB record found for '{title}' — using fallback description)")

    params = {**AUTH, "title": title, "description": desc_text}
    status, data = get(f"{BASE}/jobsworth", params)

    print(f"  TITLE: {title}  (DB: {db_title})")
    print(f"  Status: {status}  |  Keys: {list(data.keys()) if isinstance(data, dict) else '?'}")
    print(f"  Raw JSON:")
    print(json.dumps(data, indent=4))
    print()
    time.sleep(1)

conn.close()
print(SEP)
print("Diagnostic complete.")
