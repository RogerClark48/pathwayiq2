"""
extract_provider_urls.py
Finds the "APPLY NOW" link on each gmiot.ac.uk course page and stages the
extracted provider URL in the url_updates table for review before any changes
are made to gmiot_courses.

Run from project root with venv active:
    venv/Scripts/python.exe scripts/extract_provider_urls.py

Safe to re-run — overwrites url_updates each time (drop + recreate).
"""

import sqlite3
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "gmiot.sqlite"
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 15

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# -- Recreate staging table --------------------------------------------------
conn.execute("DROP TABLE IF EXISTS url_updates")
conn.execute("""
    CREATE TABLE url_updates (
        course_id   INTEGER PRIMARY KEY,
        old_url     TEXT,
        new_url     TEXT,
        status      TEXT   -- 'found' | 'not_found' | 'fetch_error' | 'still_gmiot'
    )
""")
conn.commit()

# -- Fetch target courses ----------------------------------------------------
rows = conn.execute(
    "SELECT course_id, provider, course_title, course_url "
    "FROM gmiot_courses "
    "WHERE course_url LIKE '%gmiot.ac.uk%' "
    "ORDER BY provider, course_id"
).fetchall()

total = len(rows)
print(f"Processing {total} gmiot.ac.uk courses\n")

n_found      = 0
n_not_found  = 0
n_fetch_err  = 0
n_still_gmiot = 0

for idx, row in enumerate(rows, start=1):
    course_id = row["course_id"]
    provider  = row["provider"]
    title     = row["course_title"]
    old_url   = row["course_url"]
    prefix    = f"[{idx}/{total}] {course_id} | {provider[:25]:<25} | {title[:35]:<35}"

    # Fetch GMIoT page
    try:
        resp = httpx.get(old_url, headers=HEADERS, timeout=TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"{prefix} | FETCH ERROR: {e}")
        conn.execute(
            "INSERT INTO url_updates VALUES (?, ?, ?, ?)",
            (course_id, old_url, None, "fetch_error"),
        )
        conn.commit()
        n_fetch_err += 1
        time.sleep(0.3)
        continue

    # Find "APPLY NOW" anchor (case-insensitive text match)
    apply_link = None
    for a in soup.find_all("a", href=True):
        if "apply now" in a.get_text(strip=True).lower():
            apply_link = a["href"].strip()
            break

    if apply_link is None:
        print(f"{prefix} | NOT FOUND")
        conn.execute(
            "INSERT INTO url_updates VALUES (?, ?, ?, ?)",
            (course_id, old_url, None, "not_found"),
        )
        conn.commit()
        n_not_found += 1
    elif "gmiot.ac.uk" in apply_link:
        print(f"{prefix} | STILL GMIOT: {apply_link}")
        conn.execute(
            "INSERT INTO url_updates VALUES (?, ?, ?, ?)",
            (course_id, old_url, apply_link, "still_gmiot"),
        )
        conn.commit()
        n_still_gmiot += 1
    else:
        print(f"{prefix} | found: {apply_link}")
        conn.execute(
            "INSERT INTO url_updates VALUES (?, ?, ?, ?)",
            (course_id, old_url, apply_link, "found"),
        )
        conn.commit()
        n_found += 1

    time.sleep(0.3)

# -- Full results table -------------------------------------------------------
print(f"\n{'course_id':<10} {'status':<12} {'provider':<30} {'title':<35} {'new_url'}")
print(f"{'-'*10} {'-'*12} {'-'*30} {'-'*35} {'-'*50}")
for r in conn.execute(
    "SELECT u.course_id, u.status, c.provider, c.course_title, u.new_url "
    "FROM url_updates u "
    "JOIN gmiot_courses c ON c.course_id = u.course_id "
    "ORDER BY u.status, c.provider, u.course_id"
):
    new = r["new_url"] or ""
    print(f"{r['course_id']:<10} {r['status']:<12} {r['provider'][:30]:<30} {r['course_title'][:35]:<35} {new}")

# -- Summary -----------------------------------------------------------------
print(f"""
{'-' * 70}
Summary
{'-' * 70}
Total processed:            {total}
Found provider URL:         {n_found}
Not found:                  {n_not_found}
Fetch errors:               {n_fetch_err}
Still gmiot.ac.uk:          {n_still_gmiot}

Results written to url_updates table in gmiot.sqlite.
Review the table, then run apply_provider_urls.py to apply changes.
""")

conn.close()
