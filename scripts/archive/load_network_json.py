"""
load_network_json.py
Load a cluster network JSON file into career_network_nodes and
career_network_edges tables in job_roles_asset.db.

Usage:
  venv/Scripts/python.exe scripts/load_network_json.py <path-to-json>

Example:
  venv/Scripts/python.exe scripts/load_network_json.py cluster_D2_network.json
"""

import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "job_roles_asset.db"


def main():
    if len(sys.argv) < 2:
        print("Usage: load_network_json.py <path-to-network-json>")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.is_absolute():
        json_path = Path.cwd() / json_path
    if not json_path.exists():
        print(f"File not found: {json_path}")
        sys.exit(1)

    network    = json.loads(json_path.read_text(encoding="utf-8"))
    cluster_id = network.get("cluster_id")
    nodes      = network.get("nodes", [])
    edges      = network.get("edges", [])

    if not cluster_id:
        print("ERROR: JSON has no cluster_id field")
        sys.exit(1)

    print(f"Loading cluster {cluster_id} — {len(nodes)} nodes, {len(edges)} edges")

    db = sqlite3.connect(DB_PATH)

    db.executescript("""
        CREATE TABLE IF NOT EXISTS career_network_nodes (
            job_id         INTEGER NOT NULL,
            cluster_id     TEXT NOT NULL,
            seniority_tier INTEGER NOT NULL,
            PRIMARY KEY (job_id, cluster_id)
        );

        CREATE TABLE IF NOT EXISTS career_network_edges (
            from_job_id INTEGER NOT NULL,
            to_job_id   INTEGER NOT NULL,
            edge_type   TEXT NOT NULL,
            cluster_id  TEXT NOT NULL,
            PRIMARY KEY (from_job_id, to_job_id, cluster_id)
        );
    """)

    # Clear existing data for this cluster before reloading
    db.execute("DELETE FROM career_network_nodes WHERE cluster_id = ?", (cluster_id,))
    db.execute("DELETE FROM career_network_edges WHERE cluster_id = ?", (cluster_id,))

    with db:
        db.executemany(
            "INSERT OR REPLACE INTO career_network_nodes (job_id, cluster_id, seniority_tier) VALUES (?,?,?)",
            [(n["job_id"], cluster_id, n["seniority_tier"]) for n in nodes],
        )
        db.executemany(
            "INSERT OR REPLACE INTO career_network_edges (from_job_id, to_job_id, edge_type, cluster_id) VALUES (?,?,?,?)",
            [(e["from_job_id"], e["to_job_id"], e["edge_type"], cluster_id) for e in edges],
        )

    n_nodes = db.execute("SELECT COUNT(*) FROM career_network_nodes WHERE cluster_id=?", (cluster_id,)).fetchone()[0]
    n_edges = db.execute("SELECT COUNT(*) FROM career_network_edges WHERE cluster_id=?", (cluster_id,)).fetchone()[0]
    db.close()

    print(f"career_network_nodes: {n_nodes} rows for cluster {cluster_id}")
    print(f"career_network_edges: {n_edges} rows for cluster {cluster_id}")
    print("Done.")


if __name__ == "__main__":
    main()
