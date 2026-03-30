"""
assign_routes.py — Assign route_id to se_occupations using Routes/{id} endpoint.
se_data.db must already exist with se_routes and se_occupations populated.
"""

import os
import sys
import time
import json
import sqlite3
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://occupational-maps-api.skillsengland.education.gov.uk/api/v1"
API_KEY = os.getenv("SKILLS_ENGLAND_API_KEY")
HEADERS = {"X-API-KEY": API_KEY}
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "se_data.db")


def api_get(path):
    url = f"{BASE_URL}{path}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json(), None
        return None, f"HTTP {r.status_code}"
    except Exception as e:
        return None, str(e)


def extract_std_codes(route_data):
    """
    Walk the route response and collect all stdCodes.
    The response may nest occupations under levels/tiers — we recurse to find them.
    Returns a list of stdCode strings.
    """
    std_codes = []

    def walk(obj):
        if isinstance(obj, dict):
            # Direct occupation record
            if "stdCode" in obj:
                std_codes.append(obj["stdCode"])
            # Recurse into all values
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(route_data)
    return list(set(std_codes))  # deduplicate


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: SKILLS_ENGLAND_API_KEY not found in environment / .env")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)

    # Load all routes
    routes = conn.execute("SELECT route_id, name FROM se_routes ORDER BY route_id").fetchall()
    print(f"Found {len(routes)} routes to process.\n")

    first = True
    total_assigned = 0

    for route_id, route_name in routes:
        data, err = api_get(f"/Routes/{route_id}")

        if err:
            print(f"  FAILED route {route_id} ({route_name}): {err}")
            time.sleep(0.2)
            continue

        # Log raw structure of first route to confirm nesting
        if first:
            print("=== Raw JSON for first route (truncated to 3000 chars) ===")
            print(json.dumps(data, indent=2)[:3000])
            print("=== End raw JSON ===\n")
            first = False

        std_codes = extract_std_codes(data)

        # Update only where route_id is still NULL (keep first assignment)
        updated = 0
        for std_code in std_codes:
            cursor = conn.execute(
                "UPDATE se_occupations SET route_id = ? WHERE std_code = ? AND route_id IS NULL",
                (route_id, std_code)
            )
            updated += cursor.rowcount

        conn.commit()
        total_assigned += updated
        print(f"  Route {route_id:>2} — {route_name:<45} {len(std_codes):>4} stdCodes found, {updated:>4} newly assigned")

        time.sleep(0.2)

    print(f"\nTotal newly assigned this run: {total_assigned}")

    # --- Summary ---
    print("\n=== Summary ===")
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(route_id) as assigned,
            COUNT(*) - COUNT(route_id) as unassigned
        FROM se_occupations
    """).fetchone()
    print(f"Total occupations : {row[0]}")
    print(f"Assigned route_id : {row[1]}")
    print(f"Unassigned        : {row[2]}")

    print("\n=== Breakdown by route ===")
    rows = conn.execute("""
        SELECT r.name, COUNT(o.std_code) as occupation_count
        FROM se_routes r
        LEFT JOIN se_occupations o ON o.route_id = r.route_id
        GROUP BY r.route_id, r.name
        ORDER BY occupation_count DESC
    """).fetchall()
    for name, count in rows:
        print(f"  {count:>4}  {name}")

    print("\n=== Coverage query ===")
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(route_id) as assigned,
            COUNT(*) - COUNT(route_id) as unassigned
        FROM se_occupations
    """).fetchone()
    print(f"total={row[0]}, assigned={row[1]}, unassigned={row[2]}")

    conn.close()
    print("\nDone.")
