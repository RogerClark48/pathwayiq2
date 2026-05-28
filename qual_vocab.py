"""
qual_vocab.py

Creates and seeds the qual_vocab table in futurefinder.sqlite.
Universal UK qualification canonical vocabulary — not institution-specific.

Usage:
  python qual_vocab.py           # create table and insert/replace all rows
  python qual_vocab.py --show    # print current table contents

Schema:
  canonical_term  TEXT PRIMARY KEY
  level_min       INTEGER  -- NULL = no fixed level
  level_max       INTEGER  -- NULL = no fixed level; equals level_min for single levels
"""

import argparse
import os
import sqlite3

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FF_DB = os.path.join(PROJECT_ROOT, "futurefinder.sqlite")

# (canonical_term, level_min, level_max)
QUAL_VOCAB = [
    ("Short Course",          None, None),
    ("Access to HE",          3,    3),
    ("A Level",               3,    3),
    ("BTEC Level 3",          3,    3),
    ("T Level",               3,    3),
    ("Apprenticeship",        3,    3),
    ("BTEC Level 4",          4,    4),
    ("CertHE",                4,    4),
    ("HNC",                   4,    4),
    ("DipHE",                 5,    5),
    ("BTEC Level 5",          5,    5),
    ("HND",                   5,    5),
    ("Foundation Degree",     5,    5),
    ("Higher Apprenticeship", 4,    5),
    ("Bachelor's Degree",     6,    6),
    ("Degree Apprenticeship", 6,    6),
    ("PgCert",                6,    7),
    ("PgDip",                 7,    7),
    ("Master's Degree",       7,    7),
    ("Doctoral Degree",       8,    8),
]


def seed(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qual_vocab (
            canonical_term  TEXT PRIMARY KEY,
            level_min       INTEGER,
            level_max       INTEGER
        )
    """)
    conn.executemany(
        "INSERT OR REPLACE INTO qual_vocab (canonical_term, level_min, level_max) VALUES (?, ?, ?)",
        QUAL_VOCAB,
    )
    conn.commit()
    print(f"qual_vocab: {len(QUAL_VOCAB)} rows written to futurefinder.sqlite")


def show(conn):
    rows = conn.execute(
        "SELECT canonical_term, level_min, level_max FROM qual_vocab ORDER BY level_min, canonical_term"
    ).fetchall()
    print(f"{'Canonical term':<25}  Level")
    print("-" * 35)
    for term, lo, hi in rows:
        if lo is None:
            level = "—"
        elif lo == hi:
            level = str(lo)
        else:
            level = f"{lo}–{hi}"
        print(f"{term:<25}  {level}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Print current table contents")
    args = parser.parse_args()

    conn = sqlite3.connect(FF_DB)
    if args.show:
        show(conn)
    else:
        seed(conn)
    conn.close()


if __name__ == "__main__":
    main()
