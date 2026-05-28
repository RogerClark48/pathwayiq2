"""
migrate_qual_type.py

Renames the raw qual_type column to qual_type_scrape, adds a new qual_type column,
and populates it with canonical values from qual_vocab.

Run once:
  python scripts/migrate_qual_type.py

Use --dry-run to preview changes without writing to the DB.
"""

import argparse
import os
import sqlite3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF_DB = os.path.join(PROJECT_ROOT, "futurefinder.sqlite")

# Maps raw scrape values → canonical qual_vocab term.
# Values not in this dict are assumed to already be canonical (1:1 mapping).
RAW_TO_CANONICAL = {
    "HTQ":                                   "HNC",
    "HTQ, HNC":                              "HNC",
    "HNC/HTQ":                               "HNC",
    "HND, HTQ":                              "HND",
    "HND/HTQ":                               "HND",
    "HNC/HND":                               "HNC",
    "BA Hons":                               "Bachelor's Degree",
    "BEng Hons":                             "Bachelor's Degree",
    "BSc Hons":                              "Bachelor's Degree",
    "FdA":                                   "Foundation Degree",
    "FdSc":                                  "Foundation Degree",
    "FdSc, Higher Apprenticeship":           "Higher Apprenticeship",
    "Higher Apprenticeship, Degree Apprenticeship": "Higher Apprenticeship",
    "Degree Apprenticeship, BSc Hons":       "Degree Apprenticeship",
    "MSc, PgDip":                            "Master's Degree",
}


def run(dry_run: bool):
    conn = sqlite3.connect(FF_DB)
    conn.row_factory = sqlite3.Row

    # Check whether migration has already been applied.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(courses)")}
    if "qual_type_scrape" in cols:
        print("Migration already applied (qual_type_scrape column exists). Nothing to do.")
        conn.close()
        return

    if "qual_type" not in cols:
        print("ERROR: qual_type column not found in courses table.")
        conn.close()
        return

    rows = conn.execute(
        "SELECT course_id, course_title, qual_type FROM courses ORDER BY course_id"
    ).fetchall()

    # Load canonical terms for validation.
    canonical_terms = {
        row[0] for row in conn.execute("SELECT canonical_term FROM qual_vocab")
    }

    changes = []
    unknowns = []
    for row in rows:
        raw = row["qual_type"]
        canonical = RAW_TO_CANONICAL.get(raw, raw)  # identity if not in map
        if canonical is not None and canonical not in canonical_terms:
            unknowns.append((row["course_id"], row["course_title"], raw, canonical))
        changes.append((row["course_id"], raw, canonical))

    print(f"Courses to update: {len(changes)}")
    print()

    if unknowns:
        print("WARNING — the following canonical values are not in qual_vocab:")
        for cid, title, raw, canon in unknowns:
            print(f"  [{cid}] {title!r:50s}  raw={raw!r}  -> {canon!r}")
        print()

    # Show all changes where raw != canonical.
    non_identity = [(cid, raw, canon) for cid, raw, canon in changes if raw != canon]
    if non_identity:
        print(f"Remapped values ({len(non_identity)}):")
        for cid, raw, canon in non_identity:
            print(f"  [{cid:3d}]  {raw!r:45s}  ->{canon!r}")
        print()

    if dry_run:
        print("Dry run — no changes written.")
        conn.close()
        return

    if unknowns:
        print("Aborting — resolve unknowns before running the migration.")
        conn.close()
        return

    # Apply migration.
    conn.execute("ALTER TABLE courses RENAME COLUMN qual_type TO qual_type_scrape")
    conn.execute("ALTER TABLE courses ADD COLUMN qual_type TEXT")

    for cid, raw, canon in changes:
        conn.execute(
            "UPDATE courses SET qual_type = ? WHERE course_id = ?",
            (canon, cid),
        )

    conn.commit()
    print(f"Migration complete. qual_type_scrape preserved; qual_type set to canonical values.")
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing to DB")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
