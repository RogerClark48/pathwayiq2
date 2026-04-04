"""
scrape_career_prospects.py

Scrapes the full career prospects / career path and progression section
from NCS and Prospects job profile pages and stores it in the
`career_prospects` column of job_roles_asset.db.

Sources:
  ncs      — <section id="CareerPathAndProgression"> on nationalcareers.service.gov.uk
  prospects — <h2>Career prospects</h2> + following siblings on prospects.ac.uk

Run from any directory:
  python scripts/scrape_career_prospects.py

Options:
  --limit N     Process at most N records (useful for testing)
  --overwrite   Re-scrape records that already have career_prospects content
"""

import argparse
import os
import sqlite3
import time

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
JOBS_DB     = os.path.join(PROJECT_DIR, "job_roles_asset.db")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 20
POLITE_DELAY    = 1.0   # seconds between requests
# ---------------------------------------------------------------------------


def extract_ncs(soup):
    section = soup.find("section", id="CareerPathAndProgression")
    if not section:
        return None
    return section.get_text(separator=" ", strip=True)


def extract_prospects(soup):
    h2 = next(
        (h for h in soup.find_all("h2") if h.get_text(strip=True) == "Career prospects"),
        None,
    )
    if not h2:
        return None
    parts = []
    for sibling in h2.find_next_siblings():
        if sibling.name == "h2":
            break
        text = sibling.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return " ".join(parts) if parts else None


def scrape(url, source):
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        return None, f"fetch error: {e}"

    soup = BeautifulSoup(r.text, "html.parser")

    if source == "ncs":
        text = extract_ncs(soup)
    elif source == "prospects":
        text = extract_prospects(soup)
    else:
        return None, f"unknown source: {source}"

    if text:
        return text, None
    return None, "section not found in page"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",     type=int, default=0,     help="Max records to process (0 = all)")
    parser.add_argument("--overwrite", action="store_true",      help="Re-scrape already-populated records")
    args = parser.parse_args()

    conn = sqlite3.connect(JOBS_DB)
    conn.row_factory = sqlite3.Row

    if args.overwrite:
        query = "SELECT id, title, source, url FROM jobs WHERE url IS NOT NULL AND url != ''"
    else:
        query = (
            "SELECT id, title, source, url FROM jobs "
            "WHERE url IS NOT NULL AND url != '' "
            "AND (career_prospects IS NULL OR career_prospects = '')"
        )

    rows = conn.execute(query).fetchall()
    if args.limit:
        rows = rows[: args.limit]

    total   = len(rows)
    success = 0
    missing = 0
    errors  = 0

    print(f"Records to process: {total}")
    print()

    for i, row in enumerate(rows, 1):
        job_id, title, source, url = row["id"], row["title"], row["source"], row["url"]
        print(f"[{i}/{total}] {title} ({source})", end=" ... ", flush=True)

        text, err = scrape(url, source)

        if text:
            conn.execute(
                "UPDATE jobs SET career_prospects = ? WHERE id = ?",
                (text, job_id),
            )
            conn.commit()
            print(f"OK ({len(text)} chars)")
            success += 1
        elif err and "section not found" in err:
            print(f"MISSING — {err}")
            missing += 1
        else:
            print(f"ERROR — {err}")
            errors += 1

        if i < total:
            time.sleep(POLITE_DELAY)

    conn.close()

    print()
    print(f"Done — success: {success} | section missing: {missing} | errors: {errors}")


if __name__ == "__main__":
    main()
