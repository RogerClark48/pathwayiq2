"""
validate_clusters_haiku.py
Validate connections.db cluster assignments against Haiku's judgement.
20 sampled jobs across all clusters. Read-only.
"""

import sqlite3
import json
import os
import sys
from dotenv import load_dotenv
import anthropic

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

JOBS_DB        = r"C:\Dev\pathwayiq2\job_roles_asset.db"
CONNECTIONS_DB = r"C:\Dev\pathwayiq2\connections.db"
MODEL          = "claude-haiku-4-5-20251001"

SAMPLE_JOB_IDS = [
    1248,   # web developer              – cluster 1  – HIGH
    973,    # game developer             – cluster 2  – esports probe
    50,     # ai engineer                – cluster 3  – HIGH
    963,    # forensic computer analyst  – cluster 4  – HIGH
    469,    # network engineer           – cluster 5  – HIGH
    94,     # building control officer   – cluster 6  – MEDIUM
    98,     # building technician        – cluster 7  – HIGH
    390,    # land surveyor              – cluster 8  – SPARSE
    980,    # geotechnical engineer      – cluster 9  – SPARSE
    418,    # marine engineer            – cluster 10 – MEDIUM
    146,    # climate scientist          – cluster 11 – SPARSE
    1085,   # nuclear engineer           – cluster 12 – MEDIUM
    1050,   # materials engineer         – cluster 13 – SPARSE
    95,     # building services engineer – cluster 14 – HIGH
    1042,   # maintenance engineer       – cluster 15b – MEDIUM
    235,    # electrical engineer        – cluster 16 – HIGH
    249,    # engineering operative      – cluster 17 – MEDIUM
    608,    # robotics engineer          – cluster 18 – digital-to-cluster-18 probe
    209,    # digital delivery manager   – cluster 19 – SPARSE
    417,    # manufacturing systems eng  – cluster 20 – HIGH
]

SYSTEM_PROMPT = (
    "You are a careers expert. You will be given a job title and a list of career clusters, "
    "each with their member job roles. Your task is to identify which clusters this job "
    "genuinely belongs to or naturally connects to as a career destination. Be strict — only "
    "include clusters where there is a clear, direct career relationship. "
    'Return a JSON array of cluster_ids only. Example: ["1", "5"]'
)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    db  = sqlite3.connect(JOBS_DB)
    cdb = sqlite3.connect(CONNECTIONS_DB)
    db.row_factory  = sqlite3.Row
    cdb.row_factory = sqlite3.Row

    # All clusters with their member job titles
    clusters = {}
    for row in db.execute("SELECT cluster_id, name FROM clusters ORDER BY cluster_id"):
        clusters[row["cluster_id"]] = {"name": row["name"], "members": []}

    for row in db.execute(
        "SELECT cluster_id, normalized_title FROM jobs WHERE cluster_id IS NOT NULL ORDER BY cluster_id, id"
    ):
        cid = row["cluster_id"]
        if cid in clusters:
            clusters[cid]["members"].append(row["normalized_title"])

    # Sampled jobs
    jobs = {}
    placeholders = ",".join("?" * len(SAMPLE_JOB_IDS))
    for row in db.execute(
        f"SELECT id, normalized_title, cluster_id FROM jobs WHERE id IN ({placeholders})",
        SAMPLE_JOB_IDS,
    ):
        jobs[row["id"]] = {
            "id": row["id"],
            "title": row["normalized_title"],
            "assigned_cluster": row["cluster_id"],
        }

    # For each sample job: which cluster_ids does it appear alongside in connections.db?
    # Method: find all courses that connect to this job, then collect the distinct
    # cluster_ids of ALL jobs connected to those same courses.
    cdb.execute(f"ATTACH DATABASE '{JOBS_DB}' AS jdb")

    job_cluster_reach = {}
    for jid in SAMPLE_JOB_IDS:
        rows = cdb.execute(
            """
            SELECT DISTINCT jdb.jobs.cluster_id
            FROM course_job_connections cjc
            JOIN course_job_connections cjc2 ON cjc2.course_id = cjc.course_id
            JOIN jdb.jobs ON jdb.jobs.id = cjc2.job_id
            WHERE cjc.job_id = ?
              AND jdb.jobs.cluster_id IS NOT NULL
            """,
            (jid,),
        ).fetchall()
        job_cluster_reach[jid] = sorted(set(r[0] for r in rows))

    db.close()
    cdb.close()
    return jobs, clusters, job_cluster_reach


# ── Haiku call ────────────────────────────────────────────────────────────────

def build_cluster_context(clusters):
    lines = []
    for cid, data in sorted(clusters.items(), key=lambda x: x[0]):
        members = " | ".join(data["members"])
        lines.append(f"  {cid}: {data['name']} — {members}")
    return "\n".join(lines)


def ask_haiku(client, job_title, cluster_context):
    user_msg = (
        f"Job title: {job_title}\n\n"
        f"Career clusters:\n{cluster_context}\n\n"
        "Which cluster_ids does this job genuinely connect to?"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    # Extract JSON array
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        print(f"  WARNING: unexpected Haiku output: {raw!r}", file=sys.stderr)
        return []
    return json.loads(raw[start:end])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    client = anthropic.Anthropic()
    jobs, clusters, job_cluster_reach = load_data()
    cluster_context = build_cluster_context(clusters)

    noise_tally = {}   # cluster_id → count (in connections but not Haiku)
    gap_tally   = {}   # cluster_id → count (in Haiku but not connections)

    results = []

    for jid in SAMPLE_JOB_IDS:
        job = jobs.get(jid)
        if not job:
            print(f"WARNING: job {jid} not found", file=sys.stderr)
            continue

        print(f"  Calling Haiku for: {job['title']} (id {jid})...", file=sys.stderr)
        haiku_ids = set(ask_haiku(client, job["title"], cluster_context))
        conn_ids  = set(job_cluster_reach.get(jid, []))

        noise = sorted(conn_ids - haiku_ids)
        gaps  = sorted(haiku_ids - conn_ids)

        for c in noise:
            noise_tally[c] = noise_tally.get(c, 0) + 1
        for c in gaps:
            gap_tally[c] = gap_tally.get(c, 0) + 1

        results.append({
            "id":          jid,
            "title":       job["title"],
            "assigned":    job["assigned_cluster"],
            "haiku":       sorted(haiku_ids),
            "connections": sorted(conn_ids),
            "noise":       noise,
            "gaps":        gaps,
        })

    # ── Print report ──────────────────────────────────────────────────────────

    def cluster_label(cid):
        c = clusters.get(cid, {})
        return f"{cid}: {c.get('name', '?')}"

    SEP = "=" * 100

    print("\n" + SEP)
    print("CLUSTER VALIDATION REPORT — Haiku vs connections.db")
    print(SEP)

    for r in results:
        print(f"\nJob {r['id']:4d} | {r['title']}")
        print(f"  Assigned cluster : {cluster_label(r['assigned'])}")
        print(f"  Haiku says       : {', '.join(cluster_label(c) for c in r['haiku']) or '(none)'}")
        print(f"  Connections shows: {', '.join(cluster_label(c) for c in r['connections']) or '(none)'}")
        if r["noise"]:
            print(f"  [NOISE] In conn, not Haiku : {', '.join(cluster_label(c) for c in r['noise'])}")
        else:
            print(f"  [OK] No noise")
        if r["gaps"]:
            print(f"  [GAP]  In Haiku, not conn  : {', '.join(cluster_label(c) for c in r['gaps'])}")
        else:
            print(f"  [OK] No gaps")

    print("\n" + SEP)
    print("SUMMARY ACROSS ALL 20 JOBS")
    print(SEP)

    print("\nMost frequent NOISE clusters (in connections but not Haiku — potential false positives):")
    if noise_tally:
        for cid, cnt in sorted(noise_tally.items(), key=lambda x: -x[1]):
            print(f"  {cnt:2}x  {cluster_label(cid)}")
    else:
        print("  (none)")

    print("\nMost frequent GAP clusters (in Haiku but not connections — potential missing links):")
    if gap_tally:
        for cid, cnt in sorted(gap_tally.items(), key=lambda x: -x[1]):
            print(f"  {cnt:2}x  {cluster_label(cid)}")
    else:
        print("  (none)")

    print()


if __name__ == "__main__":
    main()
