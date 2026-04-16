"""
cluster_pathway_pilot.py
Haiku inference pass across four pilot clusters.
Describes each cluster as a career territory with natural tracks and progression.
"""

import sqlite3
import sys
from dotenv import load_dotenv
import anthropic

load_dotenv()

JOBS_DB     = r"C:\Dev\pathwayiq2\job_roles_asset.db"
OUTPUT_FILE = r"C:\Dev\pathwayiq2\scripts\cluster_pathway_pilot_output.txt"
MODEL       = "claude-haiku-4-5-20251001"

PILOT_CLUSTER_IDS = ["5", "9", "11", "18"]

SYSTEM_PROMPT = (
    "You are a career guidance specialist. You will be given a cluster of related job roles "
    "from the engineering and technology sector. Your task is to describe this as a career "
    "territory — what kind of work it involves, what natural pathways or tracks exist within "
    "it, and how roles relate to each other in terms of progression. Be specific about the "
    "jobs named. Do not impose a structure that isn't there — if the cluster has one clear "
    "ladder, say so; if it has parallel tracks that don't connect, say that instead. "
    "Write in plain prose, not bullet points."
)


def load_cluster(db, cluster_id):
    cluster = db.execute(
        "SELECT cluster_id, name FROM clusters WHERE cluster_id = ?", (cluster_id,)
    ).fetchone()
    if not cluster:
        return None, []

    jobs = db.execute(
        """
        SELECT title, level, adzuna_salary_estimate, entry_routes, progression
        FROM jobs
        WHERE cluster_id = ?
        ORDER BY level, title
        """,
        (cluster_id,),
    ).fetchall()

    return cluster, jobs


def build_user_message(cluster, jobs):
    cluster_name = cluster["name"]

    # Jobs table: title / level / salary
    job_lines = []
    for j in jobs:
        parts = [j["title"]]
        if j["level"] is not None:
            parts.append(f"RQF {j['level']}")
        if j["adzuna_salary_estimate"] is not None:
            parts.append(f"~£{int(j['adzuna_salary_estimate']):,}")
        job_lines.append(" / ".join(parts))

    # Entry routes + progression per job
    detail_lines = []
    for j in jobs:
        entry      = (j["entry_routes"] or "").strip()
        progression = (j["progression"] or "").strip()
        if entry or progression:
            detail_lines.append(f"{j['title']}")
            if entry:
                detail_lines.append(f"  Entry routes: {entry}")
            if progression:
                detail_lines.append(f"  Progression: {progression}")

    msg = (
        f"Cluster: {cluster_name}\n\n"
        f"Jobs (title / RQF level / estimated salary):\n"
        + "\n".join(f"  {l}" for l in job_lines)
        + "\n\nFor each job, here is what the source data says about entry routes and progression:\n"
        + "\n".join(detail_lines)
        + "\n\nDescribe this cluster as a career territory. Identify any natural tracks or "
        "sub-pathways, and indicate how a person might move through it from entry level to "
        "senior roles."
    )
    return msg


def main():
    db = sqlite3.connect(JOBS_DB)
    db.row_factory = sqlite3.Row
    client = anthropic.Anthropic()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for cluster_id in PILOT_CLUSTER_IDS:
            cluster, jobs = load_cluster(db, cluster_id)
            if not cluster:
                print(f"WARNING: cluster {cluster_id} not found", file=sys.stderr)
                continue

            print(f"  Calling Haiku for cluster {cluster_id}: {cluster['name']}...", file=sys.stderr)

            user_msg = build_user_message(cluster, jobs)

            response = client.messages.create(
                model=MODEL,
                max_tokens=2500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )

            narrative = response.content[0].text.strip()

            out.write("=" * 40 + "\n")
            out.write(f"CLUSTER {cluster_id}: {cluster['name']}\n")
            out.write("=" * 40 + "\n\n")
            out.write(narrative)
            out.write("\n\n\n")

            print(f"    Done ({response.usage.output_tokens} tokens)", file=sys.stderr)

    db.close()
    print(f"\nOutput written to: {OUTPUT_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
