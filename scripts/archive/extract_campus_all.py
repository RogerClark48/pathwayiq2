"""
extract_campus_all.py
Extracts delivery campus/location from each GMIoT course source page and
writes the result to campus_name in gmiot_courses.

Run from project root with venv active:
    venv/Scripts/python.exe scripts/extract_campus_all.py

Safe to re-run — skips records where campus_name IS NOT NULL.
"""

import os
import sqlite3
import time
from pathlib import Path

import anthropic
import httpx
from bs4 import BeautifulSoup

BASE_DIR   = Path(__file__).parent.parent
DB_PATH    = BASE_DIR / "gmiot.sqlite"
HAIKU      = "claude-haiku-4-5-20251001"
HEADERS    = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
TIMEOUT    = 15
TEXT_LIMIT = 3000

SYSTEM_PROMPT = (
    "You extract delivery location information from course web pages. "
    "Return only the location string (e.g. 'Stretford Campus', 'ATC Building') "
    "or the word null. Nothing else."
)

USER_PROMPT = (
    "Does this course page mention a specific campus, building name, or delivery location? "
    "Examples: 'Stretford Campus', 'ATC Building', 'Peel Park Campus', 'Ancoats'. "
    "If yes, return just the location name — as short as possible. "
    "If only the college or university name is mentioned with no specific campus or building, return null. "
    "Return only the location string or the word null — nothing else.\n\n{text}"
)

client = anthropic.Anthropic()

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT course_id, course_title, provider, course_url "
    "FROM gmiot_courses ORDER BY provider, course_id"
).fetchall()

total = len(rows)
n_extracted = 0
n_null      = 0
n_skipped   = 0
n_failed    = 0

print(f"Processing {total} courses\n")

for idx, row in enumerate(rows, start=1):
    course_id = row["course_id"]
    title     = row["course_title"]
    provider  = row["provider"]
    url       = row["course_url"]
    prefix    = f"[{idx}/{total}] {course_id} | {provider[:25]:<25} | {title[:40]:<40}"

    # Skip if no URL
    if not url:
        print(f"{prefix} | SKIP (no URL)")
        n_skipped += 1
        continue

    # Skip if already extracted
    existing = conn.execute(
        "SELECT campus_name FROM gmiot_courses WHERE course_id = ?", (course_id,)
    ).fetchone()
    if existing and existing["campus_name"] is not None:
        print(f"{prefix} | SKIP (already done: {existing['campus_name']})")
        n_skipped += 1
        continue

    # Fetch page
    try:
        resp = httpx.get(url, headers=HEADERS, timeout=TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)[:TEXT_LIMIT]
    except Exception as e:
        print(f"{prefix} | FETCH ERROR: {e}")
        n_failed += 1
        continue

    # Ask Haiku
    try:
        msg = client.messages.create(
            model=HAIKU,
            max_tokens=50,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": USER_PROMPT.format(text=text)}],
        )
        raw = msg.content[0].text.strip()
        campus = None if raw.lower() == "null" else raw
    except Exception as e:
        print(f"{prefix} | HAIKU ERROR: {e}")
        n_failed += 1
        continue

    # Write result
    conn.execute(
        "UPDATE gmiot_courses SET campus_name = ? WHERE course_id = ?",
        (campus, course_id),
    )
    conn.commit()

    if campus:
        print(f"{prefix} | extracted: {campus}")
        n_extracted += 1
    else:
        print(f"{prefix} | null")
        n_null += 1

    time.sleep(0.3)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"""
{'-' * 80}
Summary
{'-' * 80}
Extracted:                  {n_extracted}
Null (no location found):   {n_null}
Skipped (no URL or done):   {n_skipped}
Failed (fetch/API error):   {n_failed}
""")

# ── By provider ──────────────────────────────────────────────────────────────
print("Results by provider:")
for prov_row in conn.execute(
    "SELECT provider, "
    "SUM(CASE WHEN campus_name IS NOT NULL THEN 1 ELSE 0 END) as extracted, "
    "SUM(CASE WHEN campus_name IS NULL THEN 1 ELSE 0 END) as null_count "
    "FROM gmiot_courses GROUP BY provider ORDER BY provider"
):
    print(f"  {prov_row[0]}: {prov_row[1]} extracted, {prov_row[2]} null")

# ── Extracted values grouped by provider ─────────────────────────────────────
print("\nExtracted campus_name values by provider:")
print(f"  {'Provider':<40} {'campus_name':<35} {'Count'}")
print(f"  {'-'*40} {'-'*35} {'-'*5}")
for r in conn.execute(
    "SELECT provider, campus_name, COUNT(*) as course_count "
    "FROM gmiot_courses "
    "GROUP BY provider, campus_name "
    "ORDER BY provider, campus_name"
):
    campus_display = r[1] if r[1] is not None else "(null)"
    print(f"  {r[0]:<40} {campus_display:<35} {r[2]}")

conn.close()
