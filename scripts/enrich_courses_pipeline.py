"""
enrich_courses_pipeline.py
Definitive enrichment pipeline for any course table that follows the gmiot_courses schema.

Reads rows where overview IS NULL, fetches each course_url, sends page text to Sonnet,
and writes five fields back to the table:
    overview, what_you_will_learn, entry_requirements, progression, campus_name

Usage:
    venv/Scripts/python.exe scripts/enrich_courses_pipeline.py
    venv/Scripts/python.exe scripts/enrich_courses_pipeline.py --db path/to/other.sqlite --table other_courses
    venv/Scripts/python.exe scripts/enrich_courses_pipeline.py --test 3

Arguments:
    --db      Path to SQLite database (default: gmiot.sqlite in project root)
    --table   Table name to enrich (default: gmiot_courses)
    --test N  Process only the first N unenriched rows

Safe to re-run — skips rows where overview IS NOT NULL.
"""

import argparse
import sqlite3
import time
from pathlib import Path

import anthropic
import httpx
from bs4 import BeautifulSoup

BASE_DIR   = Path(__file__).parent.parent
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich a course table from provider URLs using Sonnet"
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to SQLite database (default: gmiot.sqlite in project root)"
    )
    parser.add_argument(
        "--table", default="gmiot_courses",
        help="Table name to enrich (default: gmiot_courses)"
    )
    parser.add_argument(
        "--test", type=int, metavar="N",
        help="Process only the first N unenriched rows"
    )
    parser.add_argument(
        "--rerun-thin", action="store_true",
        help="Re-run courses where any field contains 'Not specified on this page' "
             "or 'Insufficient information on this page'"
    )
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else BASE_DIR / "gmiot.sqlite"
    table   = args.table

    client = anthropic.Anthropic()
    conn   = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if args.rerun_thin:
        result = conn.execute(
            f"""UPDATE {table} SET
                    overview            = NULL,
                    what_you_will_learn = NULL,
                    entry_requirements  = NULL,
                    progression         = NULL,
                    campus_name         = NULL
                WHERE overview            LIKE '%Not specified on this page%'
                   OR overview            LIKE '%Insufficient information on this page%'
                   OR what_you_will_learn LIKE '%Not specified on this page%'
                   OR entry_requirements  LIKE '%Not specified on this page%'
                   OR progression         LIKE '%Not specified on this page%'"""
        )
        conn.commit()
        print(f"Cleared {result.rowcount} thin records for re-enrichment\n")

    rows = conn.execute(
        f"SELECT course_id, course_title, provider, course_url "
        f"FROM {table} "
        f"WHERE overview IS NULL AND course_url IS NOT NULL "
        f"ORDER BY provider, course_id"
    ).fetchall()

    if args.test:
        rows = rows[:args.test]
        print(f"Test mode: processing first {args.test} unenriched rows\n")

    total  = len(rows)
    counts = {"done": 0, "fetch_error": 0, "no_content": 0, "api_error": 0}

    print(f"Enriching {total} courses in {table} from {db_path.name}\n")

    for idx, row in enumerate(rows, start=1):
        course_id = row["course_id"]
        title     = row["course_title"]
        provider  = row["provider"]
        url       = row["course_url"]
        prefix    = f"[{idx}/{total}] {course_id} | {provider[:25]:<25} | {title[:38]:<38}"

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
            f"""UPDATE {table} SET
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

    print(f"\n{'-' * 70}")
    print(f"Done: {counts['done']}  |  Fetch errors: {counts['fetch_error']}  |  "
          f"No content: {counts['no_content']}  |  API errors: {counts['api_error']}")

    if counts["done"] > 0:
        print(f"\nResults by provider:")
        print(f"  {'Provider':<35} {'done':>5} {'null overview':>14}")
        print(f"  {'-'*35} {'-'*5} {'-'*14}")
        for r in conn.execute(
            f"""SELECT provider,
                       SUM(CASE WHEN overview IS NOT NULL THEN 1 ELSE 0 END) as done,
                       SUM(CASE WHEN overview IS NULL     THEN 1 ELSE 0 END) as pending
                FROM {table} GROUP BY provider ORDER BY provider"""
        ):
            print(f"  {r[0]:<35} {r[1]:>5} {r[2]:>14}")

    conn.close()


if __name__ == "__main__":
    main()
