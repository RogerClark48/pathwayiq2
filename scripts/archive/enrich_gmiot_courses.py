"""
enrich_gmiot_courses.py
Enriches gmiot_courses_raw records via Claude Sonnet API (tool use).
Writes to gmiot_courses table in gmiot.sqlite.

Usage:
    python enrich_gmiot_courses.py            # enrich all unenriched records
    python enrich_gmiot_courses.py --test 3   # test mode: enrich first N records only
    python enrich_gmiot_courses.py --id 12    # re-enrich a specific course_id
"""

import argparse
import sqlite3
import time
from pathlib import Path

import anthropic

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "gmiot.sqlite"

MODEL      = "claude-sonnet-4-5"
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS gmiot_courses (
    course_id           INTEGER PRIMARY KEY,
    course_title        TEXT NOT NULL,
    provider            TEXT NOT NULL,
    subject_area        TEXT,
    level               INTEGER,
    qual_type           TEXT,
    mode                TEXT,
    course_url          TEXT,
    ssa_code            TEXT,
    ssa_label           TEXT,
    esco_code           TEXT,
    overview            TEXT,
    what_you_will_learn TEXT,
    entry_requirements  TEXT,
    progression         TEXT
);
"""

INSERT_SQL = """
INSERT INTO gmiot_courses (
    course_id, course_title, provider, subject_area, level, qual_type,
    mode, course_url, ssa_code, ssa_label, esco_code,
    overview, what_you_will_learn, entry_requirements, progression
) VALUES (
    :course_id, :course_title, :provider, :subject_area, :level, :qual_type,
    :mode, :course_url, :ssa_code, :ssa_label, NULL,
    :overview, :what_you_will_learn, :entry_requirements, :progression
);
"""

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

ENRICH_TOOL = {
    "name": "enrich_course",
    "description": "Return enriched content and SSA classification for a course",
    "input_schema": {
        "type": "object",
        "properties": {
            "ssa_code": {
                "type": "string",
                "description": "SSA code from the controlled taxonomy (1-15)",
            },
            "ssa_label": {
                "type": "string",
                "description": "Full SSA label text matching the code",
            },
            "overview": {
                "type": "string",
                "description": "2-sentence course overview for prospective students",
            },
            "what_you_will_learn": {
                "type": "string",
                "description": "4-5 bullet points of key topics, one per line, hyphen prefix",
            },
            "entry_requirements": {
                "type": "string",
                "description": "2-4 bullet points of entry requirements, one per line, hyphen prefix",
            },
            "progression": {
                "type": "string",
                "description": "2-3 bullet points of progression routes, one per line, hyphen prefix",
            },
        },
        "required": [
            "ssa_code",
            "ssa_label",
            "overview",
            "what_you_will_learn",
            "entry_requirements",
            "progression",
        ],
    },
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a course data specialist writing concise, plain-English course
summaries for a student-facing web application. Each summary helps a
prospective student aged 16-25 decide whether a course is right for them.

You will be given a course title, provider, qualification type, level,
and subject area. Return structured content using the enrich_course tool.

Length limits are strict:
- overview: 2 sentences maximum
- what_you_will_learn: 4-5 bullet points, one line each
- entry_requirements: 2-4 bullet points, one line each
- progression: 2-3 bullet points, one line each

Format bullet lists with a hyphen and space prefix on each line.
Write for clarity, not marketing. Avoid phrases like "cutting-edge",
"world-class", or "exciting opportunities".
Adapt content to the qualification type — a T Level description should
read differently to an MSc.

SSA taxonomy (assign the single best-fit code and label):
1  - Health, Public Services and Care
2  - Science and Mathematics
3  - Agriculture, Horticulture and Animal Care
4  - Engineering and Manufacturing Technologies
5  - Construction, Planning and the Built Environment
6  - Information and Communication Technology
7  - Retail and Commercial Enterprise
8  - Leisure, Travel and Tourism
9  - Arts, Media and Publishing
10 - History, Philosophy and Theology
11 - Social Sciences
12 - Languages, Literature and Culture
13 - Education and Training
14 - Preparation for Life and Work
15 - Business, Administration and Law
"""


def build_user_prompt(row: dict) -> str:
    level_str = str(row["level"]) if row["level"] is not None else "Not specified"
    return (
        f"Course title: {row['course_title']}\n"
        f"Provider: {row['provider']}\n"
        f"Qualification type: {row['qual_type'] or 'Not specified'}\n"
        f"Level: {level_str}\n"
        f"GM IoT subject area: {row['subject_area'] or 'Not specified'}\n\n"
        "Assign the best-fit SSA code and label from the controlled taxonomy.\n"
        "Write the four content sections for the detail view."
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def enrich_course(client: anthropic.Anthropic, row: dict) -> dict:
    """Call Claude API and return the tool use input dict."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[ENRICH_TOOL],
        tool_choice={"type": "tool", "name": "enrich_course"},
        messages=[{"role": "user", "content": build_user_prompt(row)}],
    )

    # With forced tool_choice the first content block is always a tool_use block
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return tool_block.input


# ---------------------------------------------------------------------------
# Verification queries
# ---------------------------------------------------------------------------

def run_verification(cur: sqlite3.Cursor) -> None:
    print("\n--- Verification ---")

    total = cur.execute("SELECT COUNT(*) FROM gmiot_courses").fetchone()[0]
    print(f"\nTotal records in gmiot_courses: {total}")

    print("\nSSA distribution:")
    print(f"  {'Code':<6} {'Label':<45} Count")
    print(f"  {'-'*6} {'-'*45} -----")
    for code, label, count in cur.execute(
        "SELECT ssa_code, ssa_label, COUNT(*) FROM gmiot_courses "
        "GROUP BY ssa_code, ssa_label ORDER BY CAST(ssa_code AS INTEGER)"
    ):
        print(f"  {code:<6} {label:<45} {count}")

    print("\nSpot check — course_id 1:")
    row = cur.execute("SELECT * FROM gmiot_courses WHERE course_id = 1").fetchone()
    if row:
        cols = [d[0] for d in cur.description]
        for col, val in zip(cols, row):
            display = val if val is not None else "NULL"
            if isinstance(display, str) and "\n" in display:
                print(f"  {col}:")
                for line in display.splitlines():
                    print(f"    {line}")
            else:
                print(f"  {col}: {display}")
    else:
        print("  (no record with course_id = 1)")

    null_count = cur.execute(
        "SELECT COUNT(*) FROM gmiot_courses "
        "WHERE overview IS NULL OR what_you_will_learn IS NULL "
        "OR entry_requirements IS NULL OR progression IS NULL"
    ).fetchone()[0]
    print(f"\nRecords with NULL content fields: {null_count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich GMIoT courses via Claude API")
    parser.add_argument("--test", type=int, metavar="N",
                        help="Test mode: process only the first N unenriched records")
    parser.add_argument("--id", type=int, dest="course_id",
                        help="Re-enrich a single course_id (must be deleted from gmiot_courses first)")
    args = parser.parse_args()

    client = anthropic.Anthropic()

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.executescript(CREATE_SQL)
    con.commit()

    # Fetch raw records
    if args.course_id is not None:
        rows = cur.execute(
            "SELECT * FROM gmiot_courses_raw WHERE course_id = ?", (args.course_id,)
        ).fetchall()
    else:
        rows = cur.execute("SELECT * FROM gmiot_courses_raw ORDER BY course_id").fetchall()

    total_raw = len(rows)

    if args.test:
        # In test mode, skip already-enriched records first, then cap at N
        unenriched = [
            r for r in rows
            if not cur.execute(
                "SELECT 1 FROM gmiot_courses WHERE course_id = ?", (r["course_id"],)
            ).fetchone()
        ]
        rows = unenriched[: args.test]
        print(f"Test mode: processing {len(rows)} record(s)\n")
    else:
        print(f"Processing up to {total_raw} record(s)\n")

    enriched  = 0
    skipped   = 0
    failed_ids: list[int] = []

    for idx, row in enumerate(rows, start=1):
        course_id    = row["course_id"]
        course_title = row["course_title"]
        label        = f"[{idx}/{len(rows)}] {course_title}"

        # Skip if already enriched (unless --id was used with prior delete)
        if cur.execute(
            "SELECT 1 FROM gmiot_courses WHERE course_id = ?", (course_id,)
        ).fetchone():
            print(f"{label} — skipped")
            skipped += 1
            continue

        try:
            result = enrich_course(client, dict(row))

            cur.execute(INSERT_SQL, {
                "course_id":           course_id,
                "course_title":        row["course_title"],
                "provider":            row["provider"],
                "subject_area":        row["subject_area"],
                "level":               row["level"],
                "qual_type":           row["qual_type"],
                "mode":                row["mode"],
                "course_url":          row["course_url"],
                "ssa_code":            result["ssa_code"],
                "ssa_label":           result["ssa_label"],
                "overview":            result["overview"],
                "what_you_will_learn": result["what_you_will_learn"],
                "entry_requirements":  result["entry_requirements"],
                "progression":         result["progression"],
            })
            con.commit()
            print(f"{label} — done  (SSA {result['ssa_code']})")
            enriched += 1

            if args.test:
                print(f"\n  SSA:                {result['ssa_code']} — {result['ssa_label']}")
                print(f"\n  overview:")
                for line in result["overview"].splitlines():
                    print(f"    {line}")
                print(f"\n  what_you_will_learn:")
                for line in result["what_you_will_learn"].splitlines():
                    print(f"    {line}")
                print(f"\n  entry_requirements:")
                for line in result["entry_requirements"].splitlines():
                    print(f"    {line}")
                print(f"\n  progression:")
                for line in result["progression"].splitlines():
                    print(f"    {line}")
                print()

        except Exception as exc:
            print(f"{label} — FAILED: {exc}")
            failed_ids.append(course_id)

        if idx < len(rows):
            time.sleep(0.5)

    # Summary
    total_enriched = cur.execute("SELECT COUNT(*) FROM gmiot_courses").fetchone()[0]
    print(f"\nRun complete")
    print(f"  Enriched this run : {enriched}")
    print(f"  Skipped           : {skipped}")
    print(f"  Failed            : {len(failed_ids)}")
    print(f"  Total in table    : {total_enriched}")

    if failed_ids:
        print(f"\nFailed course IDs (re-run individually with --id <n>):")
        for fid in failed_ids:
            print(f"  {fid}")

    if not args.test:
        run_verification(cur)

    con.close()


if __name__ == "__main__":
    main()
