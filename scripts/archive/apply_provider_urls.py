"""
apply_provider_urls.py
Applies staged URL updates from url_updates to gmiot_courses.url.

Run ONLY after reviewing the url_updates table from extract_provider_urls.py.

Run from project root with venv active:
    venv/Scripts/python.exe scripts/apply_provider_urls.py
"""

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "gmiot.sqlite"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# Verify staging table exists
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
if "url_updates" not in tables:
    print("ERROR: url_updates table not found. Run extract_provider_urls.py first.")
    conn.close()
    raise SystemExit(1)

pending = conn.execute(
    "SELECT course_id, old_url, new_url FROM url_updates WHERE status = 'found'"
).fetchall()

if not pending:
    print("No rows with status='found' in url_updates. Nothing to apply.")
    conn.close()
    raise SystemExit(0)

print(f"Applying {len(pending)} URL updates to gmiot_courses...\n")

n_updated = 0
for row in pending:
    conn.execute(
        "UPDATE gmiot_courses SET course_url = ? WHERE course_id = ?",
        (row["new_url"], row["course_id"]),
    )
    print(f"  {row['course_id']} -> {row['new_url']}")
    n_updated += 1

conn.commit()

skipped = conn.execute(
    "SELECT COUNT(*) FROM url_updates WHERE status != 'found'"
).fetchone()[0]

print(f"""
{'-' * 70}
Updated:  {n_updated} courses
Skipped:  {skipped} (not_found / fetch_error / still_gmiot)

url_updates table preserved for reference.
""")

conn.close()
