"""
pull_se_typical_titles.py — Add typical_job_titles to se_occupations in se_data.db.
Calls GET /Occupations/{stdCode}?expand=occupation.typicaljobtitles for each occupation.
Safe to re-run — skips rows where typical_job_titles IS NOT NULL.
"""

import os
import sys
import time
import sqlite3
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL  = "https://occupational-maps-api.skillsengland.education.gov.uk/api/v1"
API_KEY   = os.getenv("SKILLS_ENGLAND_API_KEY")
HEADERS   = {"X-API-KEY": API_KEY}
ROOT      = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH   = os.path.join(ROOT, "se_data.db")


def ensure_column(conn):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(se_occupations)").fetchall()]
    if "typical_job_titles" not in cols:
        conn.execute("ALTER TABLE se_occupations ADD COLUMN typical_job_titles TEXT")
        conn.commit()
        print("Added typical_job_titles column.")
    else:
        print("typical_job_titles column already exists.")


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: SKILLS_ENGLAND_API_KEY not set in environment / .env")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    ensure_column(conn)

    # Only process rows not yet populated
    rows = conn.execute(
        "SELECT std_code FROM se_occupations WHERE typical_job_titles IS NULL ORDER BY std_code"
    ).fetchall()
    std_codes = [r[0] for r in rows]
    total = len(std_codes)
    print(f"{total} occupations to process (skipping already-populated rows).\n")

    n_updated = 0
    n_null    = 0   # call succeeded but no titles returned
    n_failed  = 0   # HTTP error
    batch_updated = 0
    batch_failed  = 0

    for i, std_code in enumerate(std_codes, 1):
        url = f"{BASE_URL}/Occupations/{std_code}?expand=occupation.typicaljobtitles"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                # Titles are at root level: [{name: "...", isGreen: false}, ...]
                raw    = data.get("typicalJobTitles") or []
                titles = [item["name"].strip() for item in raw if item.get("name")]

                if titles:
                    value = "|".join(titles)
                    conn.execute(
                        "UPDATE se_occupations SET typical_job_titles = ? WHERE std_code = ?",
                        (value, std_code)
                    )
                    n_updated += 1
                    batch_updated += 1
                else:
                    # Successful call but no titles
                    n_null += 1

            elif r.status_code == 404:
                n_failed += 1
                batch_failed += 1
            else:
                print(f"  WARN {std_code}: HTTP {r.status_code}", flush=True)
                n_failed += 1
                batch_failed += 1

        except Exception as e:
            print(f"  ERROR {std_code}: {e}", flush=True)
            n_failed += 1
            batch_failed += 1

        if i % 50 == 0:
            conn.commit()
            print(f"[{i}/{total}] {batch_updated} updated, {batch_failed} failed", flush=True)
            batch_updated = 0
            batch_failed  = 0

        time.sleep(0.1)

    conn.commit()

    # Final summary
    remaining = total - (i % 50 if i % 50 else 50)  # leftover after last batch print
    print(f"\n=== Complete ===")
    print(f"Processed : {total}")
    print(f"Updated   : {n_updated}  (titles written)")
    print(f"Null      : {n_null}    (call succeeded, no titles)")
    print(f"Failed    : {n_failed}  (HTTP error or exception)")

    # Verify
    populated = conn.execute(
        "SELECT COUNT(*) FROM se_occupations WHERE typical_job_titles IS NOT NULL"
    ).fetchone()[0]
    print(f"\nTotal rows with typical_job_titles : {populated}/951")

    # Sample
    print("\nSample (5 rows):")
    for std_code, name, titles in conn.execute(
        "SELECT std_code, name, typical_job_titles FROM se_occupations "
        "WHERE typical_job_titles IS NOT NULL LIMIT 5"
    ).fetchall():
        print(f"  {std_code}  {name}")
        print(f"    -> {titles}")

    conn.close()
    print("\nDone.")
