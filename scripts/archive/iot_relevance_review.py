"""
iot_relevance_review.py — Review results of the IoT relevance tagging pass.

Prints:
  1. Count by relevance level, broken down by RQF level
  2. Random sample of 25 'high' records

Run:
    C:\Dev\pathwayiq2\venv\Scripts\python.exe scripts/iot_relevance_review.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "job_roles_asset.db"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# ---------------------------------------------------------------------------
# 1. Count by iot_relevant × RQF level
# ---------------------------------------------------------------------------
print("=" * 60)
print("COUNTS BY RELEVANCE LEVEL × RQF LEVEL")
print("=" * 60)

cur.execute("""
    SELECT
        iot_relevant,
        level,
        COUNT(*) AS n
    FROM jobs
    GROUP BY iot_relevant, level
    ORDER BY
        CASE iot_relevant
            WHEN 'high'   THEN 1
            WHEN 'medium' THEN 2
            WHEN 'low'    THEN 3
            WHEN 'no'     THEN 4
            WHEN 'error'  THEN 5
            ELSE 6
        END,
        level
""")
rows = cur.fetchall()

current_relevance = None
relevance_total = 0
for row in rows:
    if row["iot_relevant"] != current_relevance:
        if current_relevance is not None:
            print(f"  {'TOTAL':<12} {relevance_total}")
            print()
        current_relevance = row["iot_relevant"]
        relevance_total = 0
        label = row["iot_relevant"] if row["iot_relevant"] else "NULL (no content)"
        print(f"{label.upper()}")
    level_label = f"L{row['level']}" if row["level"] else "level NULL"
    print(f"  {level_label:<12} {row['n']}")
    relevance_total += row["n"]

if current_relevance is not None:
    print(f"  {'TOTAL':<12} {relevance_total}")
    print()

# Overall totals
cur.execute("""
    SELECT iot_relevant, COUNT(*) AS n
    FROM jobs
    GROUP BY iot_relevant
    ORDER BY
        CASE iot_relevant
            WHEN 'high'   THEN 1
            WHEN 'medium' THEN 2
            WHEN 'low'    THEN 3
            WHEN 'no'     THEN 4
            WHEN 'error'  THEN 5
            ELSE 6
        END
""")
totals = cur.fetchall()
print("=" * 60)
print("OVERALL TOTALS")
print("=" * 60)
for row in totals:
    label = row["iot_relevant"] if row["iot_relevant"] else "NULL (no content)"
    print(f"  {label:<12} {row['n']}")
print()

# ---------------------------------------------------------------------------
# 2. Sample of 25 'high' records
# ---------------------------------------------------------------------------
print("=" * 60)
print("SAMPLE — 25 RANDOM 'HIGH' RECORDS")
print("=" * 60)

cur.execute("""
    SELECT id, title, source, level, iot_relevance_note
    FROM jobs
    WHERE iot_relevant = 'high'
    ORDER BY RANDOM()
    LIMIT 25
""")
sample = cur.fetchall()

for i, row in enumerate(sample, 1):
    level = f"L{row['level']}" if row["level"] else "L?"
    source = row["source"] or ""
    note = row["iot_relevance_note"] or ""
    print(f"{i:2d}. [{level} {source:8s}] {row['title']}")
    print(f"      {note}")

print()

# ---------------------------------------------------------------------------
# 3. Sample of 25 'medium' records
# ---------------------------------------------------------------------------
print("=" * 60)
print("SAMPLE — 25 RANDOM 'MEDIUM' RECORDS")
print("=" * 60)

cur.execute("""
    SELECT id, title, source, level, iot_relevance_note
    FROM jobs
    WHERE iot_relevant = 'medium'
    ORDER BY RANDOM()
    LIMIT 25
""")
sample = cur.fetchall()

for i, row in enumerate(sample, 1):
    level = f"L{row['level']}" if row["level"] else "L?"
    source = row["source"] or ""
    note = row["iot_relevance_note"] or ""
    print(f"{i:2d}. [{level} {source:8s}] {row['title']}")
    print(f"      {note}")

conn.close()
