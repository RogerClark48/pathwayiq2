"""
apply_all_enrichment.py
Applies staged enrichment results from url_updates to gmiot_courses.

Run ONLY after reviewing the url_updates table populated by enrich_all_courses.py.

Run from project root with venv active:
    venv/Scripts/python.exe scripts/apply_all_enrichment.py
"""

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "gmiot.sqlite"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# Verify staging table and enrichment columns exist
tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if "url_updates" not in tables:
    print("ERROR: url_updates table not found. Run enrich_all_courses.py first.")
    conn.close()
    raise SystemExit(1)

cols = {row[1] for row in conn.execute("PRAGMA table_info(url_updates)")}
required = {"overview", "what_you_will_learn", "entry_requirements", "progression",
            "campus_name", "enrichment_status"}
missing = required - cols
if missing:
    print(f"ERROR: url_updates is missing columns: {missing}")
    print("Run enrich_all_courses.py first to populate the staging table.")
    conn.close()
    raise SystemExit(1)

pending = conn.execute(
    """SELECT u.course_id, u.new_url, u.status,
              u.overview, u.what_you_will_learn, u.entry_requirements,
              u.progression, u.campus_name,
              c.course_title, c.provider
       FROM url_updates u
       JOIN gmiot_courses c ON c.course_id = u.course_id
       WHERE u.enrichment_status = 'done'
       ORDER BY c.provider, u.course_id"""
).fetchall()

if not pending:
    print("No rows with enrichment_status='done' in url_updates. Nothing to apply.")
    conn.close()
    raise SystemExit(0)

print(f"Applying enrichment for {len(pending)} courses to gmiot_courses...\n")

n_updated       = 0
n_url_updated   = 0

for row in pending:
    course_id  = row["course_id"]
    new_url    = row["new_url"]
    url_status = row["status"]   # 'found' | 'existing' | etc.
    title      = row["course_title"]
    provider   = row["provider"]

    # Determine whether to update the URL:
    # Only replace if the staged URL came from url extraction (status='found')
    current_url = conn.execute(
        "SELECT course_url FROM gmiot_courses WHERE course_id = ?", (course_id,)
    ).fetchone()["course_url"]

    update_url = new_url if (url_status == "found" and new_url) else current_url

    conn.execute(
        """UPDATE gmiot_courses SET
               course_url          = ?,
               overview            = ?,
               what_you_will_learn = ?,
               entry_requirements  = ?,
               progression         = ?,
               campus_name         = ?
           WHERE course_id = ?""",
        (
            update_url,
            row["overview"],
            row["what_you_will_learn"],
            row["entry_requirements"],
            row["progression"],
            row["campus_name"],
            course_id,
        ),
    )

    url_flag = " (URL updated)" if (url_status == "found" and new_url and new_url != current_url) else ""
    print(f"  {course_id:>3} | {provider[:30]:<30} | {title[:40]:<40}{url_flag}")
    n_updated += 1
    if url_status == "found" and new_url and new_url != current_url:
        n_url_updated += 1

conn.commit()

skipped = conn.execute(
    "SELECT COUNT(*) FROM url_updates WHERE enrichment_status != 'done'"
).fetchone()[0]

print(f"""
{'-' * 70}
Courses updated:        {n_updated}
  of which URL changed: {n_url_updated}
Skipped (not done):     {skipped}

url_updates table preserved for reference.
""")

# -- Provider summary --------------------------------------------------------
print("Update summary by provider:")
print(f"  {'Provider':<35} {'updated':>8}")
print(f"  {'-'*35} {'-'*8}")
for r in conn.execute(
    """SELECT c.provider, COUNT(*) as cnt
       FROM gmiot_courses c
       JOIN url_updates u ON u.course_id = c.course_id
       WHERE u.enrichment_status = 'done'
       GROUP BY c.provider ORDER BY c.provider"""
):
    print(f"  {r[0]:<35} {r[1]:>8}")

conn.close()
