"""
pull_se_data.py — Skills England Occupational Maps data pull
Pulls all routes, occupations, and progression pairs into se_data.db.
One-off pre-pass; no live API calls at runtime.
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


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS se_routes (
            route_id    INTEGER PRIMARY KEY,
            name        TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS se_occupations (
            std_code    TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            level       INTEGER,
            route_id    INTEGER,
            FOREIGN KEY (route_id) REFERENCES se_routes(route_id)
        );

        CREATE TABLE IF NOT EXISTS se_progressions (
            std_code_from   TEXT NOT NULL,
            std_code_to     TEXT NOT NULL,
            PRIMARY KEY (std_code_from, std_code_to)
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(path):
    """GET request. Returns (response_json, None) on success or (None, error_msg) on failure."""
    url = f"{BASE_URL}{path}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json(), None
        else:
            return None, f"HTTP {r.status_code}"
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Step 1 — Routes
# ---------------------------------------------------------------------------

def pull_routes(conn):
    print("\n=== Step 1: Pulling routes ===")
    data, err = api_get("/Routes")
    if err:
        print(f"ERROR fetching routes: {err}")
        sys.exit(1)

    routes = [(r["routeId"], r["name"]) for r in data]
    conn.executemany(
        "INSERT OR REPLACE INTO se_routes (route_id, name) VALUES (?, ?)",
        routes
    )
    conn.commit()

    for route_id, name in routes:
        print(f"  [{route_id}] {name}")
    print(f"Stored {len(routes)} routes.")
    return {r["routeId"]: r["name"] for r in data}


# ---------------------------------------------------------------------------
# Step 2 — Occupation list
# ---------------------------------------------------------------------------

def pull_occupation_list():
    print("\n=== Step 2: Pulling occupation list ===")
    data, err = api_get("/Occupations")
    if err:
        print(f"ERROR fetching occupations: {err}")
        sys.exit(1)

    approved = [o for o in data if o.get("status") == 1]
    print(f"Total returned: {len(data)} — Approved (status=1): {len(approved)}")
    return approved


# ---------------------------------------------------------------------------
# Step 3 — Progression neighbourhood per occupation
# ---------------------------------------------------------------------------

def upsert_occupation(conn, std_code, name, level, route_id=None):
    conn.execute(
        """INSERT INTO se_occupations (std_code, name, level, route_id)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(std_code) DO UPDATE SET
               name = excluded.name,
               level = excluded.level,
               route_id = COALESCE(excluded.route_id, se_occupations.route_id)""",
        (std_code, name, level, route_id)
    )


def pull_progressions(conn, approved_occupations):
    print("\n=== Step 3: Pulling progression neighbourhoods ===")
    total = len(approved_occupations)
    failed = []

    for i, occ in enumerate(approved_occupations, 1):
        std_code = occ["stdCode"]

        data, err = api_get(f"/OccupationalProgression/{std_code}")

        if err:
            print(f"  FAILED {std_code}: {err}")
            failed.append((std_code, err))
            time.sleep(0.2)
            continue

        # Upsert all occupations in the neighbourhood
        for o in data.get("occupations", []):
            upsert_occupation(conn, o["stdCode"], o["name"], o.get("level"))

        # Insert all progression pairs (ignore duplicates via PRIMARY KEY)
        pairs = [
            (p["stdCodeFrom"], p["stdCodeTo"])
            for p in data.get("progressions", [])
        ]
        if pairs:
            conn.executemany(
                "INSERT OR IGNORE INTO se_progressions (std_code_from, std_code_to) VALUES (?, ?)",
                pairs
            )

        conn.commit()

        if i % 50 == 0:
            print(f"  Processed {i}/{total}...")

        time.sleep(0.2)

    print(f"Completed {total} occupations. Failures: {len(failed)}")
    return failed


# ---------------------------------------------------------------------------
# Step 4 — Summary report
# ---------------------------------------------------------------------------

def summary_report(conn, failed):
    print("\n=== Summary ===")

    total_occ = conn.execute("SELECT COUNT(*) FROM se_occupations").fetchone()[0]
    total_prog = conn.execute("SELECT COUNT(*) FROM se_progressions").fetchone()[0]
    print(f"Occupations stored : {total_occ}")
    print(f"Progression pairs  : {total_prog}")

    if failed:
        print(f"\nFailed stdCodes ({len(failed)}):")
        for std_code, reason in failed:
            print(f"  {std_code}: {reason}")
    else:
        print("No failures.")

    print("\nSample progression chains (5):")
    rows = conn.execute("""
        SELECT o1.name, o1.level, o2.name, o2.level
        FROM se_progressions p
        JOIN se_occupations o1 ON p.std_code_from = o1.std_code
        JOIN se_occupations o2 ON p.std_code_to = o2.std_code
        LIMIT 5
    """).fetchall()
    for from_name, from_level, to_name, to_level in rows:
        print(f"  L{from_level} {from_name}  →  L{to_level} {to_name}")


# ---------------------------------------------------------------------------
# Verification queries
# ---------------------------------------------------------------------------

def verification_queries(conn):
    print("\n=== Verification queries ===")

    print("\n-- Total counts --")
    print("Occupations:", conn.execute("SELECT COUNT(*) FROM se_occupations").fetchone()[0])
    print("Progressions:", conn.execute("SELECT COUNT(*) FROM se_progressions").fetchone()[0])

    print("\n-- Occupations by level --")
    for level, count in conn.execute(
        "SELECT level, COUNT(*) FROM se_occupations GROUP BY level ORDER BY level"
    ).fetchall():
        print(f"  Level {level}: {count}")

    print("\n-- Sample progression chains (20) --")
    rows = conn.execute("""
        SELECT o1.name, o1.level, o2.name, o2.level
        FROM se_progressions p
        JOIN se_occupations o1 ON p.std_code_from = o1.std_code
        JOIN se_occupations o2 ON p.std_code_to = o2.std_code
        LIMIT 20
    """).fetchall()
    for from_name, from_level, to_name, to_level in rows:
        print(f"  L{from_level} {from_name}  →  L{to_level} {to_name}")

    print("\n-- Occupations with no progression (isolated nodes) --")
    count = conn.execute("""
        SELECT COUNT(*) FROM se_occupations o
        WHERE NOT EXISTS (
            SELECT 1 FROM se_progressions p
            WHERE p.std_code_from = o.std_code OR p.std_code_to = o.std_code
        )
    """).fetchone()[0]
    print(f"  Isolated: {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: SKILLS_ENGLAND_API_KEY not found in environment / .env")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    pull_routes(conn)
    approved_occupations = pull_occupation_list()
    failed = pull_progressions(conn, approved_occupations)
    summary_report(conn, failed)
    verification_queries(conn)

    conn.close()
    print("\nDone. se_data.db written.")
