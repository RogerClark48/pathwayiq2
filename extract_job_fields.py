"""
extract_job_fields.py
Extracts structured fields from enriched_description blobs in emiot_jobs_asset.db.
Adds five named columns to the jobs table: overview, typical_duties,
skills_required, entry_routes, progression.

Inspection findings (recorded here for reference):
- Source blob is enriched_description, not description
- Primary key is id, not job_id
- 1,252 total records; ~1,216 have enriched_description; ~36 do not (skipped)
- Two blob formats coexist:
    Format A: "SECTION_NAME: content", bullets as corrupted UTF-8 characters
    Format B: "## SECTION_NAME" headers, clean hyphen-space bullets
- salary field is prose narrative from SALARY RANGE section
- progression field covers career development only (CAREER PROGRESSION section)
- Encoding artefacts present in Format A blobs (corrupted bullet/pound characters)

# ============================================================
# NOTE: TWO-FORMAT HANDLING IS TEMPORARY — EMIOT DATA ONLY
# ============================================================
# The dual-format complexity (Format A / Format B) exists solely
# because emiot_jobs_asset.db was built across multiple enrichment
# passes with inconsistent output formatting.
#
# When the GMIoT jobs database is built, it will use a single
# consistent format from the start. At that point this script
# (or its GMIoT equivalent) should be simplified to handle
# one format only, and this note should be removed.
# ============================================================

Usage:
    python extract_job_fields.py             # extract all unprocessed records
    python extract_job_fields.py --test 5    # test mode: show N records, no commit
    python extract_job_fields.py --id 42     # re-extract single record (clear fields first)
"""

import argparse
import sqlite3
import time
from pathlib import Path

import anthropic

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "emiot_jobs_asset.db"

MODEL      = "claude-sonnet-4-5"
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# New columns
# ---------------------------------------------------------------------------

ALTER_STATEMENTS = [
    "ALTER TABLE jobs ADD COLUMN overview TEXT",
    "ALTER TABLE jobs ADD COLUMN typical_duties TEXT",
    "ALTER TABLE jobs ADD COLUMN skills_required TEXT",
    "ALTER TABLE jobs ADD COLUMN entry_routes TEXT",
    "ALTER TABLE jobs ADD COLUMN salary TEXT",
    "ALTER TABLE jobs ADD COLUMN progression TEXT",
]

UPDATE_SQL = """
UPDATE jobs
SET overview        = :overview,
    typical_duties  = :typical_duties,
    skills_required = :skills_required,
    entry_routes    = :entry_routes,
    salary          = :salary,
    progression     = :progression
WHERE id = :id
"""

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

EXTRACT_TOOL = {
    "name": "extract_job_fields",
    "description": "Extract structured content fields from a job description blob",
    "input_schema": {
        "type": "object",
        "properties": {
            "overview": {
                "type": "string",
                "description": "2-sentence prose summary of the role",
            },
            "typical_duties": {
                "type": "string",
                "description": "4-6 bullet points of day-to-day duties, hyphen-space prefix, one per line",
            },
            "skills_required": {
                "type": "string",
                "description": "4-6 bullet points of required skills, hyphen-space prefix, one per line",
            },
            "entry_routes": {
                "type": "string",
                "description": "2-4 bullet points of entry routes, hyphen-space prefix, one per line",
            },
            "salary": {
                "type": "string",
                "description": "Narrative salary information from the blob — typical ranges, career-stage breakdowns, banding. 1-3 sentences of prose.",
            },
            "progression": {
                "type": "string",
                "description": "2-3 bullet points of career development and advancement routes only, hyphen-space prefix, one per line",
            },
        },
        "required": [
            "overview",
            "typical_duties",
            "skills_required",
            "entry_routes",
            "salary",
            "progression",
        ],
    },
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a structured data extraction specialist. You will be given a job
description in blob format with section headers. Extract the content into
five named fields using the extract_job_fields tool.

The blob may use one of two formats:
- Format A: "SECTION NAME: content" on the same line, with bullet points
  using corrupted UTF-8 characters (ÔÇó, •, or similar) — treat these as bullets
- Format B: "## SECTION NAME" headers on their own line, with hyphen-space bullets

Extract faithfully from the source — do not invent or add content not
present in the blob. If a section is missing or thin, use what is there.

The salary field should capture the narrative salary information from the SALARY RANGE
section as prose — preserve career-stage breakdowns, banding, and specific figures.
The progression field should cover career development and advancement only, drawn
from the CAREER PROGRESSION section.

Format all bullet list fields with a hyphen and space prefix on each line.
Correct any encoding artefacts silently:
  ÔÇó → hyphen-space bullet
  Â£ → £
  • or ÿ or similar corrupted bullet characters → hyphen-space bullet

The overview field should be 2 sentences of prose summarising the role.
"""


def build_user_prompt(row: dict) -> str:
    return (
        f"Job title: {row['title']}\n"
        f"Source: {row['source']}\n\n"
        f"Description:\n{row['enriched_description']}\n\n"
        "Extract the five structured fields from this description."
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def extract_fields(client: anthropic.Anthropic, row: dict) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_job_fields"},
        messages=[{"role": "user", "content": build_user_prompt(row)}],
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return tool_block.input


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_columns_if_missing(cur: sqlite3.Cursor) -> None:
    existing = {row[1] for row in cur.execute("PRAGMA table_info(jobs)").fetchall()}
    for stmt in ALTER_STATEMENTS:
        col = stmt.split("ADD COLUMN ")[1].split()[0]
        if col not in existing:
            cur.execute(stmt)
            print(f"  Added column: {col}")


def print_extracted(result: dict) -> None:
    for field in ("overview", "typical_duties", "skills_required", "entry_routes", "salary", "progression"):
        print(f"\n  {field}:")
        for line in result[field].splitlines():
            print(f"    {line}")
    print()


def run_verification(cur: sqlite3.Cursor) -> None:
    print("\n--- Verification ---")

    row = cur.execute("""
        SELECT COUNT(*) as total,
               COUNT(overview) as has_overview,
               COUNT(typical_duties) as has_duties,
               COUNT(skills_required) as has_skills,
               COUNT(entry_routes) as has_entry,
               COUNT(progression) as has_progression
        FROM jobs
    """).fetchone()
    print(f"\n  Total records  : {row[0]}")
    print(f"  has overview   : {row[1]}")
    print(f"  has duties     : {row[2]}")
    print(f"  has skills     : {row[3]}")
    print(f"  has entry      : {row[4]}")
    print(f"  has progression: {row[5]}")

    print("\n  Source distribution:")
    for source, total, extracted in cur.execute("""
        SELECT source, COUNT(*) as total, COUNT(overview) as extracted
        FROM jobs GROUP BY source
    """):
        print(f"    {source:<12} total={total}  extracted={extracted}")

    print("\n  Encoding artefact check (should be 0):")
    artefacts = cur.execute("""
        SELECT COUNT(*) FROM jobs
        WHERE overview LIKE '%ÔÇó%' OR overview LIKE '%Â£%'
           OR typical_duties LIKE '%ÔÇó%' OR typical_duties LIKE '%Â£%'
    """).fetchone()[0]
    print(f"    Records with artefacts: {artefacts}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract job fields from enriched_description blobs")
    parser.add_argument("--test", type=int, nargs="?", const=0, metavar="N",
                        help="Test mode: print results without committing. Omit N to auto-select one Format A and one Format B record.")
    parser.add_argument("--id", type=int, dest="record_id",
                        help="Re-extract a single record by id (clear its fields first)")
    args = parser.parse_args()

    client = anthropic.Anthropic()

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Add new columns if not already present
    add_columns_if_missing(cur)
    con.commit()

    # Fetch records to process
    if args.record_id is not None:
        rows = cur.execute(
            "SELECT id, title, source, enriched_description FROM jobs WHERE id = ?",
            (args.record_id,)
        ).fetchall()
    elif args.test:
        # Test mode without --id: pick one Format B (## headers) and one Format A (SECTION:) record
        fmt_b = cur.execute(
            "SELECT id, title, source, enriched_description FROM jobs "
            "WHERE enriched_description LIKE '# %\n\n## %' "
            "AND overview IS NULL LIMIT 1"
        ).fetchall()
        fmt_a = cur.execute(
            "SELECT id, title, source, enriched_description FROM jobs "
            "WHERE enriched_description LIKE 'OVERVIEW:%' "
            "AND overview IS NULL LIMIT 1"
        ).fetchall()
        rows = fmt_b + fmt_a
        if not rows:
            # Fall back to first N unprocessed if no unprocessed examples of each format
            limit = args.test if args.test else 2
            rows = cur.execute(
                "SELECT id, title, source, enriched_description FROM jobs "
                "WHERE overview IS NULL AND enriched_description IS NOT NULL "
                "ORDER BY id LIMIT ?", (limit,)
            ).fetchall()
    else:
        rows = cur.execute(
            "SELECT id, title, source, enriched_description FROM jobs "
            "WHERE overview IS NULL AND enriched_description IS NOT NULL "
            "ORDER BY id"
        ).fetchall()

    # Count records skipped due to no enriched_description (for reporting)
    no_blob_count = cur.execute(
        "SELECT COUNT(*) FROM jobs WHERE enriched_description IS NULL"
    ).fetchone()[0]

    if args.test:
        rows = list(rows)[: args.test]
        print(f"Test mode: processing {len(rows)} record(s) (no commit)\n")
    else:
        print(f"Records to process : {len(rows)}")
        print(f"No enriched_description (skipped): {no_blob_count}\n")

    extracted  = 0
    skipped    = 0
    failed_ids: list[int] = []

    for idx, row in enumerate(rows, start=1):
        record_id    = row["id"]
        title        = row["title"]
        source       = row["source"]
        label        = f"[{idx}/{len(rows)}] {title} ({source})"

        # Skip if already extracted (unless --id used after manual clear)
        if row["enriched_description"] is None:
            print(f"{label} — no blob, skipping")
            skipped += 1
            continue

        already_done = cur.execute(
            "SELECT overview FROM jobs WHERE id = ?", (record_id,)
        ).fetchone()
        if already_done and already_done[0] is not None:
            print(f"{label} — skipped")
            skipped += 1
            continue

        try:
            result = extract_fields(client, dict(row))

            if args.test:
                blob = row["enriched_description"] or ""
                fmt = "Format B (## headers)" if blob.startswith("# ") else "Format A (SECTION: headers)"
                print(f"{label} — done  [{fmt}]")
                print_extracted(result)
            else:
                cur.execute(UPDATE_SQL, {
                    "id":               record_id,
                    "overview":         result["overview"],
                    "typical_duties":   result["typical_duties"],
                    "skills_required":  result["skills_required"],
                    "entry_routes":     result["entry_routes"],
                    "salary":           result["salary"],
                    "progression":      result["progression"],
                })
                con.commit()
                print(f"{label} — done")
            extracted += 1

        except Exception as exc:
            print(f"{label} — FAILED: {exc}")
            failed_ids.append(record_id)

        if idx < len(rows):
            time.sleep(0.5)

    # Summary
    print(f"\nRun complete")
    print(f"  Extracted this run : {extracted}")
    print(f"  Skipped            : {skipped}")
    print(f"  No blob (excluded) : {no_blob_count}")
    print(f"  Failed             : {len(failed_ids)}")

    if failed_ids:
        print(f"\nFailed IDs (re-run with --id <n> after clearing fields):")
        for fid in failed_ids:
            print(f"  {fid}")

    if not args.test:
        run_verification(cur)

    con.close()


if __name__ == "__main__":
    main()
