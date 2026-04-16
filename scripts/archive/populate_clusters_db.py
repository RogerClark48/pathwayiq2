"""
populate_clusters_db.py — Add clusters table and cluster_id column to job_roles_asset.db,
then populate from cluster JSON files and cluster_analysis.txt.

Safe to re-run: uses INSERT OR REPLACE for clusters, UPDATE for jobs.

Run:
    C:\Dev\pathwayiq2\venv\Scripts\python.exe scripts/populate_clusters_db.py
"""

import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "job_roles_asset.db"
ANALYSIS_TXT = PROJECT_ROOT / "cluster_analysis.txt"

# ---------------------------------------------------------------------------
# Cluster metadata — id, name, JSON file (narrative comes from _narrative.txt)
# cluster 15 is intentionally absent — superseded by 15a and 15b
# ---------------------------------------------------------------------------
CLUSTERS = [
    ("1",   "Software development"),
    ("2",   "Games & creative digital"),
    ("3",   "AI & data science"),
    ("4",   "Cyber security"),
    ("5",   "IT systems & infrastructure"),
    ("6",   "Building surveying & compliance"),
    ("7",   "Construction & civil engineering"),
    ("8",   "GIS & land surveying"),
    ("9",   "Earth & geo sciences"),
    ("10",  "Maritime & naval"),
    ("11",  "Aviation, space & weather"),
    ("12",  "Medical physics & nuclear"),
    ("13",  "Life sciences, chemistry & materials"),
    ("14",  "Energy & renewables"),
    ("15a", "Agricultural & land-based engineering"),
    ("15b", "Plant & maintenance engineering"),
    ("16",  "Electrical, electronics & telecoms"),
    ("17",  "Precision & metalwork manufacturing"),
    ("18",  "Mechanical, aerospace & automotive"),
    ("19",  "IT project & production management"),
    ("20",  "Chemical & process engineering"),
]

# 15a members (post-hoc split from cluster 15 — not in cluster_analysis.txt)
CLUSTER_15A_TITLES = {
    "agricultural engineer",
    "agricultural engineering technician",
    "land-based engineer",
}

# ---------------------------------------------------------------------------
# Step 1 — Parse Ward medium membership from cluster_analysis.txt
# Returns dict: cluster_id (str) -> set of lowercase display titles
# Cluster 15 is split into 15a / 15b here.
# ---------------------------------------------------------------------------
def parse_ward_medium(path: Path) -> dict[str, set[str]]:
    membership: dict[str, set[str]] = {}
    in_ward = in_medium = False
    current_id = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.rstrip()
            if "METHOD A: WARD LINKAGE" in s:
                in_ward = True; continue
            if in_ward and "METHOD B:" in s:
                break
            if not in_ward:
                continue
            if "--- Medium" in s:
                in_medium = True; continue
            if in_medium and s.startswith("--- "):
                break
            if not in_medium:
                continue
            if s.startswith("Cluster "):
                try:
                    current_id = str(int(s.split()[1]))
                except (IndexError, ValueError):
                    current_id = None
                continue
            if current_id and s.startswith("  ") and s.strip():
                title_lower = s.strip().lower()
                # Split cluster 15 into 15a / 15b
                if current_id == "15":
                    cid = "15a" if title_lower in CLUSTER_15A_TITLES else "15b"
                else:
                    cid = current_id
                membership.setdefault(cid, set()).add(title_lower)

    return membership

# ---------------------------------------------------------------------------
# Step 2 — Schema migration
# ---------------------------------------------------------------------------
def migrate_schema(conn: sqlite3.Connection):
    cur = conn.cursor()

    # Add cluster_id to jobs if not present
    cur.execute("PRAGMA table_info(jobs)")
    existing = {row[1] for row in cur.fetchall()}
    if "cluster_id" not in existing:
        cur.execute("ALTER TABLE jobs ADD COLUMN cluster_id TEXT")
        print("  Added column: jobs.cluster_id")
    else:
        print("  Column jobs.cluster_id already exists — skipping")

    # Create clusters table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            cluster_id   TEXT PRIMARY KEY,
            name         TEXT,
            narrative    TEXT,
            network_json TEXT,
            job_count    INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    print("  clusters table ready")

# ---------------------------------------------------------------------------
# Step 3 — Populate clusters table
# ---------------------------------------------------------------------------
def populate_clusters_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    missing_files = []

    for cluster_id, name in CLUSTERS:
        json_path      = PROJECT_ROOT / f"cluster_{cluster_id}_network.json"
        narrative_path = PROJECT_ROOT / f"cluster_{cluster_id}_narrative.txt"

        if not json_path.exists():
            print(f"  MISSING: {json_path.name}")
            missing_files.append(str(json_path.name))
            continue
        if not narrative_path.exists():
            print(f"  MISSING: {narrative_path.name}")
            missing_files.append(str(narrative_path.name))
            continue

        network_json = json_path.read_text(encoding="utf-8")
        narrative    = narrative_path.read_text(encoding="utf-8").strip()

        # Validate JSON parses cleanly
        try:
            json.loads(network_json)
        except json.JSONDecodeError as e:
            print(f"  JSON PARSE ERROR for cluster {cluster_id}: {e}")
            continue

        cur.execute("""
            INSERT OR REPLACE INTO clusters (cluster_id, name, narrative, network_json)
            VALUES (?, ?, ?, ?)
        """, (cluster_id, name, narrative, network_json))
        print(f"  Inserted cluster {cluster_id:3s} — {name}")

    conn.commit()

    if missing_files:
        print(f"\nWARNING: {len(missing_files)} file(s) missing — those clusters not inserted")
        return False
    return True

# ---------------------------------------------------------------------------
# Step 4 — Assign cluster_id to jobs
# ---------------------------------------------------------------------------
def assign_cluster_ids(conn: sqlite3.Connection, membership: dict[str, set[str]]):
    cur = conn.cursor()

    # Build lookup: lowercase_title -> cluster_id
    title_to_cluster: dict[str, str] = {}
    for cid, titles in membership.items():
        for t in titles:
            title_to_cluster[t] = cid

    # Fetch all jobs with a title
    cur.execute("SELECT id, title, normalized_title FROM jobs WHERE overview IS NOT NULL")
    all_jobs = cur.fetchall()

    matched = unmatched = 0
    unmatched_titles = []

    for job_id, title, normalized_title in all_jobs:
        # Try normalized_title first (most reliable), then display title lowercased
        nt_lower = (normalized_title or "").lower().strip()
        t_lower  = (title or "").lower().strip()

        cid = title_to_cluster.get(nt_lower) or title_to_cluster.get(t_lower)

        if cid:
            cur.execute("UPDATE jobs SET cluster_id = ? WHERE id = ?", (cid, job_id))
            matched += 1
        else:
            # Only flag if the job was tagged high-relevance — others are expected to be NULL
            cur.execute("SELECT iot_relevant FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            if row and row[0] == "high":
                unmatched_titles.append((job_id, title, normalized_title))
                unmatched += 1

    conn.commit()

    print(f"\n  Jobs matched to a cluster:   {matched}")
    print(f"  High-relevance unmatched:    {unmatched}")
    if unmatched_titles:
        print("  Unmatched high-relevance titles:")
        for jid, t, nt in sorted(unmatched_titles, key=lambda x: x[1] or ""):
            print(f"    id={jid:4d}  title={t!r}  norm={nt!r}")

    return matched, unmatched

# ---------------------------------------------------------------------------
# Step 5 — Update job_count on clusters table
# ---------------------------------------------------------------------------
def update_job_counts(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
        UPDATE clusters
        SET job_count = (
            SELECT COUNT(*) FROM jobs WHERE jobs.cluster_id = clusters.cluster_id
        )
    """)
    conn.commit()

    cur.execute("SELECT cluster_id, name, job_count FROM clusters ORDER BY cluster_id")
    rows = cur.fetchall()
    print("\n  Cluster job counts:")
    total = 0
    for cid, name, count in rows:
        print(f"    {cid:4s}  {name:<42}  {count:3d} jobs")
        total += count
    print(f"    {'':4s}  {'TOTAL':<42}  {total:3d} jobs")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Parsing Ward medium membership from cluster_analysis.txt...")
    membership = parse_ward_medium(ANALYSIS_TXT)
    for cid in sorted(membership, key=lambda x: (x.rstrip('ab'), x)):
        print(f"  Cluster {cid:4s}: {len(membership[cid])} titles")

    conn = sqlite3.connect(DB_PATH)

    print("\nMigrating schema...")
    migrate_schema(conn)

    print("\nPopulating clusters table...")
    populate_clusters_table(conn)

    print("\nAssigning cluster_id to jobs...")
    assign_cluster_ids(conn, membership)

    print("\nUpdating job_count...")
    update_job_counts(conn)

    conn.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
