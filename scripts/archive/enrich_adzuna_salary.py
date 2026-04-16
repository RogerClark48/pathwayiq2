"""
enrich_adzuna_salary.py — Enrich high-relevance job records with Adzuna
Jobsworth salary estimates and histogram-derived salary ranges.

Adds adzuna_salary_estimate, adzuna_salary_min, adzuna_salary_max, and
adzuna_queried_at columns to the jobs table. Re-run safe — skips records
where adzuna_queried_at is already set.

Run:
    C:\Dev\pathwayiq2\venv\Scripts\python.exe scripts/enrich_adzuna_salary.py
"""

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "job_roles_asset.db"
load_dotenv(PROJECT_ROOT / ".env")

APP_ID  = os.environ.get("ADZUNA_APP_ID", "")
APP_KEY = os.environ.get("ADZUNA_APP_KEY", "")

if not APP_ID or not APP_KEY:
    print("ERROR: ADZUNA_APP_ID or ADZUNA_APP_KEY not found in .env")
    raise SystemExit(1)

AUTH        = {"app_id": APP_ID, "app_key": APP_KEY}
BASE        = "https://api.adzuna.com/v1/api/jobs/gb"
DELAY       = 0.5   # seconds between jobs (2 calls per job)
MIN_COUNT   = 3     # histogram bucket count threshold

# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------
def migrate(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(jobs)")
    existing = {row[1] for row in cur.fetchall()}
    for col, typedef in [
        ("adzuna_salary_estimate", "REAL"),
        ("adzuna_salary_min",      "REAL"),
        ("adzuna_salary_max",      "REAL"),
        ("adzuna_queried_at",      "TEXT"),
    ]:
        if col not in existing:
            cur.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typedef}")
            print(f"  Added column: {col}")
        else:
            print(f"  Column {col} already exists")
    conn.commit()

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_get(url, params) -> tuple[int, dict]:
    try:
        r = requests.get(url, params=params, timeout=15)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {}
    except requests.RequestException as e:
        print(f"    NETWORK ERROR: {e}")
        return 0, {}


def jobsworth(title: str, description: str) -> float | None:
    params = {**AUTH, "title": title, "description": description[:2000]}
    status, data = api_get(f"{BASE}/jobsworth", params)
    if status == 200 and "salary" in data:
        return float(data["salary"])
    return None


def histogram_range(title: str) -> tuple[float | None, float | None]:
    params = {**AUTH, "what": title}
    status, data = api_get(f"{BASE}/histogram", params)
    if status != 200 or "histogram" not in data:
        return None, None

    hist = data["histogram"]
    # Parse and sort buckets ascending by salary value
    buckets = []
    for k, v in hist.items():
        try:
            buckets.append((float(k), int(v)))
        except (ValueError, TypeError):
            continue
    buckets.sort(key=lambda x: x[0])

    if not buckets:
        return None, None

    total = sum(c for _, c in buckets)
    if total == 0:
        return None, None

    # Use per-bucket % (not cumulative).
    THRESHOLD = 0.05
    significant = [salary for salary, count in buckets if count / total >= THRESHOLD]

    if not significant:
        return None, None

    sal_min = min(significant)

    highest = max(significant)
    # If the highest qualifying bucket is the open-ended 70k+ band, store as-is.
    # Otherwise add £10k to convert the bucket floor into the bucket ceiling.
    sal_max = highest if highest == 70000 else highest + 10000

    return sal_min, sal_max

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("Migrating schema...")
    migrate(conn)

    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, normalized_title, overview
        FROM jobs
        WHERE iot_relevant = 'high'
          AND overview IS NOT NULL
          AND adzuna_queried_at IS NULL
        ORDER BY id
    """)
    jobs = cur.fetchall()

    cur.execute("SELECT COUNT(*) FROM jobs WHERE iot_relevant = 'high' AND adzuna_queried_at IS NOT NULL")
    already_done = cur.fetchone()[0]

    print(f"\nTo enrich: {len(jobs)}  |  Already done: {already_done}\n")

    stats = {
        "total":      0,
        "jobsworth":  0,
        "histogram":  0,
        "http_errors": 0,
    }

    for i, job in enumerate(jobs, 1):
        job_id = job["id"]
        title  = job["normalized_title"] or job["title"] or ""
        desc   = job["overview"] or ""

        jw_salary   = jobsworth(title, desc)
        hist_min, hist_max = histogram_range(title)

        ts = datetime.now(timezone.utc).isoformat()
        cur.execute("""
            UPDATE jobs
            SET adzuna_salary_estimate = ?,
                adzuna_salary_min      = ?,
                adzuna_salary_max      = ?,
                adzuna_queried_at      = ?
            WHERE id = ?
        """, (jw_salary, hist_min, hist_max, ts, job_id))
        conn.commit()

        stats["total"] += 1
        if jw_salary is not None:  stats["jobsworth"]  += 1
        if hist_min  is not None:  stats["histogram"]  += 1

        jw_str   = f"£{jw_salary:,.0f}" if jw_salary is not None else "NULL"
        hist_str = f"£{hist_min:,.0f}–£{hist_max:,.0f}" if hist_min is not None else "NULL"
        print(f"[{i:3d}/{len(jobs)}] {title[:45]:<45}  JW={jw_str:<12}  Hist={hist_str}")

        if i < len(jobs):
            time.sleep(DELAY)

    conn.close()

    print(f"\nComplete.")
    print(f"  Total processed:       {stats['total']}")
    print(f"  Jobsworth returned:    {stats['jobsworth']}  (NULL: {stats['total'] - stats['jobsworth']})")
    print(f"  Histogram range:       {stats['histogram']}  (NULL: {stats['total'] - stats['histogram']})")
    print(f"  HTTP errors:           {stats['http_errors']}")

if __name__ == "__main__":
    main()
