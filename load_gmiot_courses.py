"""
load_gmiot_courses.py
Loads gmiot_courses.csv into gmiot.sqlite (gmiot_courses_raw table).
Safe to re-run — existing rows are skipped by course_id.
"""

import csv
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "gmiot.sqlite"
CSV_PATH = BASE_DIR / "gmiot_courses.csv"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS gmiot_courses_raw (
    course_id       INTEGER PRIMARY KEY,
    course_title    TEXT NOT NULL,
    provider        TEXT NOT NULL,
    subject_area    TEXT,
    level           INTEGER,
    qual_type       TEXT,
    mode            TEXT,
    course_url      TEXT,
    notes           TEXT
);
"""

INSERT_SQL = """
INSERT OR IGNORE INTO gmiot_courses_raw
    (course_id, course_title, provider, subject_area, level, qual_type, mode, course_url, notes)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

VERIFY_SQL = """
SELECT provider, COUNT(*) as course_count
FROM gmiot_courses_raw
GROUP BY provider
ORDER BY provider;
"""


def nullable(value: str):
    """Return None for blank/whitespace strings, otherwise stripped value."""
    stripped = value.strip()
    return stripped if stripped else None


def nullable_int(value: str):
    """Return None for blank strings, otherwise int."""
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.executescript(CREATE_SQL)
    con.commit()

    inserted = 0
    skipped  = 0

    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            params = (
                int(row["course_id"]),
                row["course_title"].strip(),
                row["provider"].strip(),
                nullable(row.get("subject_area", "")),
                nullable_int(row.get("level", "")),
                nullable(row.get("qual_type", "")),
                nullable(row.get("mode", "")),
                nullable(row.get("course_url", "")),
                nullable(row.get("notes", "")),
            )
            cur.execute(INSERT_SQL, params)
            if cur.rowcount == 1:
                inserted += 1
            else:
                skipped += 1

    con.commit()

    total = cur.execute("SELECT COUNT(*) FROM gmiot_courses_raw").fetchone()[0]

    print(f"\nLoad complete")
    print(f"  Inserted : {inserted}")
    print(f"  Skipped  : {skipped}")
    print(f"  Total    : {total}")

    print("\nProvider breakdown:")
    print(f"  {'Provider':<35} Count")
    print(f"  {'-'*35} -----")
    for provider, count in cur.execute(VERIFY_SQL):
        print(f"  {provider:<35} {count}")

    con.close()


if __name__ == "__main__":
    main()
