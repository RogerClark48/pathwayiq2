"""
enrich_all_courses.py
Enriches GMIoT courses from real provider pages using Sonnet.

PHASE 1 (default)
  Processes the 54 courses whose URLs were corrected (url_updates.status='found').
  Fetches new_url for each, sends page text to Sonnet, stores results in staging
  columns on url_updates. Does NOT touch gmiot_courses.

  After Phase 1 completes, review url_updates then run:
      venv/Scripts/python.exe scripts/apply_all_enrichment.py

PHASE 2 (run after apply_all_enrichment.py is complete and confirmed)
  Processes the remaining 29 courses using their existing course_url.
  Writes overview, what_you_will_learn, entry_requirements, progression, campus_name
  directly to gmiot_courses, overwriting whatever was there.

Usage:
    venv/Scripts/python.exe scripts/enrich_all_courses.py           # Phase 1
    venv/Scripts/python.exe scripts/enrich_all_courses.py --phase 2 # Phase 2

Both phases are safe to re-run:
  Phase 1 skips rows where enrichment_status = 'done' in url_updates.
  Phase 2 skips courses that already have a url_updates found row (Phase 1 territory).
"""

import argparse
import sqlite3
import time
from pathlib import Path

import anthropic
import httpx
from bs4 import BeautifulSoup

BASE_DIR   = Path(__file__).parent.parent
DB_PATH    = BASE_DIR / "gmiot.sqlite"
SONNET     = "claude-sonnet-4-6"
HEADERS    = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
TIMEOUT    = 15
TEXT_LIMIT = 10000

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

ENRICH_TOOL = {
    "name": "enrich_course",
    "description": (
        "Extract structured course content from a provider course page. "
        "Use only information explicitly present in the page text. "
        "Do not infer, assume, or invent anything not stated on the page."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "overview": {
                "type": "string",
                "description": (
                    "2-3 sentence plain-English summary of what the course covers "
                    "and who it suits. Based strictly on page content. "
                    "If the page contains insufficient information to write a summary, "
                    "write exactly: 'Insufficient information on this page.'"
                ),
            },
            "what_you_will_learn": {
                "type": "string",
                "description": (
                    "4-6 bullet points of the main topics and skills covered. "
                    "One per line, hyphen-space prefix (e.g. '- Topic'). "
                    "Use only topics explicitly named on the page. "
                    "If no topics are listed, write exactly: "
                    "'- Not specified on this page.'"
                ),
            },
            "entry_requirements": {
                "type": "string",
                "description": (
                    "3-5 bullet points of entry requirements — qualifications, grades, "
                    "experience — as stated on the page. "
                    "One per line, hyphen-space prefix. "
                    "If none are stated on the page, write exactly: "
                    "'- Not specified on this page.'"
                ),
            },
            "progression": {
                "type": "string",
                "description": (
                    "2-4 bullet points of where this course leads — further study "
                    "and employment routes as stated on the page. "
                    "One per line, hyphen-space prefix. "
                    "If none are stated, write exactly: "
                    "'- Not specified on this page.'"
                ),
            },
            "campus_name": {
                "type": ["string", "null"],
                "description": (
                    "The specific campus or building name if explicitly stated "
                    "(e.g. 'Stretford Campus', 'ATC Building'). "
                    "Return null if only the college or university name appears "
                    "or no delivery location is mentioned."
                ),
            },
        },
        "required": [
            "overview",
            "what_you_will_learn",
            "entry_requirements",
            "progression",
            "campus_name",
        ],
    },
}

SYSTEM_PROMPT = """\
You are a course data specialist extracting structured information from \
provider course pages for a student-facing career guidance tool.

IMPORTANT: Extract content only from the page text provided. \
Do not add, infer, or invent any information that is not explicitly \
present in the text. If a section is not covered on the page, say so \
rather than guessing.

Write in plain English for a prospective student aged 16-25. \
Avoid marketing language such as "cutting-edge", "world-class", \
or "exciting opportunities".\
"""

# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

STAGING_COLUMNS = [
    ("overview",            "TEXT"),
    ("what_you_will_learn", "TEXT"),
    ("entry_requirements",  "TEXT"),
    ("progression",         "TEXT"),
    ("campus_name",         "TEXT"),
    ("enrichment_status",   "TEXT"),
]


def migrate_url_updates(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(url_updates)")}
    added = []
    for col_name, col_type in STAGING_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE url_updates ADD COLUMN {col_name} {col_type}")
            added.append(col_name)
    conn.commit()
    if added:
        print(f"Added columns to url_updates: {', '.join(added)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_text(url: str) -> tuple[str | None, str | None]:
    """Return (text, None) on success or (None, error_message) on failure."""
    try:
        resp = httpx.get(url, headers=HEADERS, timeout=TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:TEXT_LIMIT], None
    except Exception as e:
        return None, str(e)


def call_sonnet(client: anthropic.Anthropic, text: str) -> dict:
    """Call Sonnet with forced tool use and return extracted fields."""
    response = client.messages.create(
        model=SONNET,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[ENRICH_TOOL],
        tool_choice={"type": "tool", "name": "enrich_course"},
        messages=[{
            "role": "user",
            "content": (
                "Extract structured course information from the following "
                "provider page text. Use only what is explicitly stated.\n\n"
                f"{text}"
            ),
        }],
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return tool_block.input


def print_summary(counts: dict) -> None:
    for k, v in counts.items():
        print(f"  {k:<20} {v}")


# ---------------------------------------------------------------------------
# Phase 1 — fetch provider pages, write to url_updates staging
# ---------------------------------------------------------------------------

def run_phase_1(conn: sqlite3.Connection, client: anthropic.Anthropic, limit: int | None = None) -> None:
    rows = conn.execute(
        """SELECT u.course_id, u.new_url, u.enrichment_status,
                  c.course_title, c.provider
           FROM url_updates u
           JOIN gmiot_courses c ON c.course_id = u.course_id
           WHERE u.status = 'found'
           ORDER BY c.provider, u.course_id"""
    ).fetchall()

    if limit:
        rows = rows[:limit]
        print(f"Test mode: processing first {limit} of 54 courses\n")

    total  = len(rows)
    counts = {"done": 0, "skipped": 0, "fetch_error": 0, "no_content": 0, "api_error": 0}

    print(f"Phase 1 — staging enrichment for {total} courses\n")

    for idx, row in enumerate(rows, start=1):
        course_id = row["course_id"]
        title     = row["course_title"]
        provider  = row["provider"]
        url       = row["new_url"]
        prefix    = f"[{idx}/{total}] {course_id} | {provider[:25]:<25} | {title[:38]:<38}"

        if row["enrichment_status"] == "done":
            print(f"{prefix} | SKIP (already done)")
            counts["skipped"] += 1
            continue

        text, err = fetch_text(url)
        if err:
            print(f"{prefix} | FETCH ERROR: {err}")
            conn.execute(
                "UPDATE url_updates SET enrichment_status='fetch_error' WHERE course_id=?",
                (course_id,)
            )
            conn.commit()
            counts["fetch_error"] += 1
            time.sleep(0.3)
            continue

        if len(text) < 200:
            print(f"{prefix} | NO CONTENT ({len(text)} chars)")
            conn.execute(
                "UPDATE url_updates SET enrichment_status='no_content' WHERE course_id=?",
                (course_id,)
            )
            conn.commit()
            counts["no_content"] += 1
            time.sleep(0.3)
            continue

        try:
            fields = call_sonnet(client, text)
        except Exception as e:
            print(f"{prefix} | SONNET ERROR: {e}")
            conn.execute(
                "UPDATE url_updates SET enrichment_status='api_error' WHERE course_id=?",
                (course_id,)
            )
            conn.commit()
            counts["api_error"] += 1
            time.sleep(0.3)
            continue

        campus = fields.get("campus_name")
        conn.execute(
            """UPDATE url_updates SET
                   overview            = ?,
                   what_you_will_learn = ?,
                   entry_requirements  = ?,
                   progression         = ?,
                   campus_name         = ?,
                   enrichment_status   = 'done'
               WHERE course_id = ?""",
            (
                fields.get("overview"),
                fields.get("what_you_will_learn"),
                fields.get("entry_requirements"),
                fields.get("progression"),
                campus,
                course_id,
            ),
        )
        conn.commit()

        campus_label = f"campus: {campus}" if campus else "campus: null"
        print(f"{prefix} | done | {campus_label}")
        counts["done"] += 1
        time.sleep(0.5)

    print(f"\n{'-' * 60}")
    print("Phase 1 summary")
    print(f"{'-' * 60}")
    print_summary(counts)

    pending_review = conn.execute(
        "SELECT COUNT(*) FROM url_updates WHERE enrichment_status = 'done'"
    ).fetchone()[0]

    if pending_review > 0:
        print(
            f"\n{pending_review} rows ready in url_updates. "
            "Review, then run:\n"
            "  venv/Scripts/python.exe scripts/apply_all_enrichment.py"
        )


# ---------------------------------------------------------------------------
# Phase 2 — fetch existing course URLs, write directly to gmiot_courses
# ---------------------------------------------------------------------------

def run_phase_2(conn: sqlite3.Connection, client: anthropic.Anthropic, limit: int | None = None) -> None:
    # All courses NOT covered by url_updates found rows
    rows = conn.execute(
        """SELECT c.course_id, c.course_title, c.provider, c.course_url
           FROM gmiot_courses c
           WHERE c.course_id NOT IN (
               SELECT course_id FROM url_updates WHERE status = 'found'
           )
           ORDER BY c.provider, c.course_id"""
    ).fetchall()

    if limit:
        rows = rows[:limit]
        print(f"Test mode: processing first {limit} of {len(rows)} remaining courses\n")

    total  = len(rows)
    counts = {"done": 0, "fetch_error": 0, "no_content": 0, "api_error": 0}

    print(f"Phase 2 — direct enrichment for {total} remaining courses\n")

    for idx, row in enumerate(rows, start=1):
        course_id = row["course_id"]
        title     = row["course_title"]
        provider  = row["provider"]
        url       = row["course_url"]
        prefix    = f"[{idx}/{total}] {course_id} | {provider[:25]:<25} | {title[:38]:<38}"

        if not url:
            print(f"{prefix} | SKIP (no URL)")
            counts["fetch_error"] += 1
            continue

        text, err = fetch_text(url)
        if err:
            print(f"{prefix} | FETCH ERROR: {err}")
            counts["fetch_error"] += 1
            time.sleep(0.3)
            continue

        if len(text) < 200:
            print(f"{prefix} | NO CONTENT ({len(text)} chars)")
            counts["no_content"] += 1
            time.sleep(0.3)
            continue

        try:
            fields = call_sonnet(client, text)
        except Exception as e:
            print(f"{prefix} | SONNET ERROR: {e}")
            counts["api_error"] += 1
            time.sleep(0.3)
            continue

        campus = fields.get("campus_name")
        conn.execute(
            """UPDATE gmiot_courses SET
                   overview            = ?,
                   what_you_will_learn = ?,
                   entry_requirements  = ?,
                   progression         = ?,
                   campus_name         = ?
               WHERE course_id = ?""",
            (
                fields.get("overview"),
                fields.get("what_you_will_learn"),
                fields.get("entry_requirements"),
                fields.get("progression"),
                campus,
                course_id,
            ),
        )
        conn.commit()

        campus_label = f"campus: {campus}" if campus else "campus: null"
        print(f"{prefix} | done | {campus_label}")
        counts["done"] += 1
        time.sleep(0.5)

    print(f"\n{'-' * 60}")
    print("Phase 2 summary")
    print(f"{'-' * 60}")
    print_summary(counts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich GMIoT courses from provider pages")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=1,
        help="1 = stage into url_updates (default); 2 = write directly to gmiot_courses"
    )
    parser.add_argument(
        "--test", type=int, metavar="N",
        help="Process only the first N courses (for testing before a full run)"
    )
    args = parser.parse_args()

    client = anthropic.Anthropic()
    conn   = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if args.phase == 1:
        migrate_url_updates(conn)
        run_phase_1(conn, client, limit=args.test)
    else:
        run_phase_2(conn, client, limit=args.test)

    conn.close()


if __name__ == "__main__":
    main()
