"""
Diagnostic script — test campus/location extraction from GMIoT course pages.
Reads first 10 TSCG (Trafford) courses, fetches each source page, and asks
Haiku to extract the delivery location.

Run from project root with venv active:
    venv/Scripts/python.exe scripts/extract_campus_test.py
"""

import os
import sqlite3
import httpx
import anthropic
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

GMIOT_DB   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gmiot.sqlite")
HAIKU      = "claude-haiku-4-5-20251001"
HEADERS    = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
TIMEOUT    = 15
TEXT_LIMIT = 3000

client = anthropic.Anthropic()

conn = sqlite3.connect(GMIOT_DB)
rows = conn.execute(
    "SELECT course_id, course_title, course_url FROM gmiot_courses "
    "WHERE provider LIKE '%Trafford%' AND course_url IS NOT NULL LIMIT 10"
).fetchall()
conn.close()

print(f"{'course_id':<12} {'extracted_location':<30} {'title'}")
print("-" * 100)

for course_id, title, url in rows:
    try:
        resp = httpx.get(url, headers=HEADERS, timeout=TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        snippet = soup.get_text(separator=" ", strip=True)[:TEXT_LIMIT]
    except Exception as e:
        print(f"{course_id:<12} FETCH ERROR: {e}")
        print(f"             URL: {url}")
        continue

    try:
        msg = client.messages.create(
            model=HAIKU,
            max_tokens=50,
            system=(
                "You extract delivery location information from course web pages. "
                "Return only the location string (e.g. 'Stretford Campus') or the word null. "
                "Nothing else."
            ),
            messages=[{
                "role": "user",
                "content": (
                    "Does this course page mention a specific campus, building name, or delivery location? "
                    "If yes, return just the location name. If no location is mentioned, return null.\n\n"
                    f"{snippet}"
                ),
            }],
        )
        location = msg.content[0].text.strip()
    except Exception as e:
        location = f"HAIKU ERROR: {e}"

    print(f"{course_id:<12} {location:<30} {title}")
