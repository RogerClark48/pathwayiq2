"""
esco_test.py — One-off ESCO API test against a sample of job titles.

Reads 25 distinct job titles from job_roles_asset.db, queries the ESCO
occupation search API for each, and writes results to esco_test_results.csv
in the project root. No database writes.

Run from any directory:
    C:\Dev\pathwayiq2\venv\Scripts\python.exe scripts/esco_test.py
"""

import csv
import sqlite3
import time
import urllib.parse
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "job_roles_asset.db"
OUTPUT_CSV = PROJECT_ROOT / "esco_test_results.csv"

# ---------------------------------------------------------------------------
# ESCO API
# ---------------------------------------------------------------------------
ESCO_SEARCH_URL = "https://ec.europa.eu/esco/api/search"
REQUEST_DELAY = 2  # seconds between requests — polite to EC's server


def query_esco(title: str) -> dict:
    """
    Query ESCO occupation search for a job title.
    Returns a dict with keys: esco_title, esco_uri, result_count.
    On error or empty results, returns safe defaults.
    """
    params = {
        "text": title,
        "type": "occupation",
        "language": "en",
        # No selectedVersion pin — let API default to current published version
    }
    url = ESCO_SEARCH_URL + "?" + urllib.parse.urlencode(params)

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"  ERROR: network/HTTP error — {e}")
        return {"esco_title": "ERROR", "esco_uri": "", "result_count": -1}
    except ValueError:
        print("  ERROR: could not parse JSON response")
        return {"esco_title": "ERROR", "esco_uri": "", "result_count": -1}

    total = data.get("total", 0)
    results = data.get("_embedded", {}).get("results", [])

    if not results:
        return {"esco_title": "", "esco_uri": "", "result_count": 0}

    top = results[0]
    return {
        "esco_title": top.get("title", ""),
        "esco_uri": top.get("uri", ""),
        "result_count": total,
    }


# ---------------------------------------------------------------------------
# Match quality heuristic
# ---------------------------------------------------------------------------
def match_quality(result_count: int) -> str:
    # Thresholds are a first-pass guess — recalibrate after reviewing the CSV
    if result_count < 0:
        return "error"
    if result_count == 0:
        return "no match"
    if result_count <= 3:
        return "strong"
    if result_count <= 10:
        return "moderate"
    return "weak"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"DB:     {DB_PATH}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT id, title, normalized_title, source
        FROM jobs
        WHERE title IS NOT NULL
        GROUP BY normalized_title
        ORDER BY RANDOM()
        LIMIT 25
    """)
    sample = cur.fetchall()
    conn.close()

    print(f"Sample size: {len(sample)} titles\n")

    rows = []
    for i, job in enumerate(sample, 1):
        job_id = job["id"]
        our_title = job["title"]          # human-readable title sent to API
        our_source = job["source"] or ""

        print(f"[{i:2d}/25] {our_title}")
        result = query_esco(our_title)
        quality = match_quality(result["result_count"])
        print(f"        → {result['esco_title'] or '(no match)'}  "
              f"(total={result['result_count']}, quality={quality})")

        rows.append({
            "job_id": job_id,
            "our_title": our_title,        # title sent to API, not normalized_title
            "our_source": our_source,
            "esco_title": result["esco_title"],
            "esco_uri": result["esco_uri"],
            "result_count": result["result_count"],
            "match_quality": quality,
        })

        if i < len(sample):
            time.sleep(REQUEST_DELAY)

    fieldnames = [
        "job_id", "our_title", "our_source",
        "esco_title", "esco_uri", "result_count", "match_quality",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
