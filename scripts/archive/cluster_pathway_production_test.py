"""
cluster_pathway_production_test.py
Single Haiku inference pass against Cluster 13 (Life sciences, chemistry & materials).
Validates the production JSON prompt before the full 21-cluster run.
"""

import sqlite3
import json
import sys
from dotenv import load_dotenv
import anthropic

load_dotenv()

JOBS_DB      = r"C:\Dev\pathwayiq2\job_roles_asset.db"
OUTPUT_JSON  = r"C:\Dev\pathwayiq2\scripts\cluster13_pathway_test_output.json"
OUTPUT_TXT   = r"C:\Dev\pathwayiq2\scripts\cluster13_pathway_test_output.txt"
MODEL        = "claude-haiku-4-5-20251001"
CLUSTER_ID   = "13"

SYSTEM_PROMPT = (
    "You are a career guidance specialist helping students aged 16 and above understand "
    "career territories in engineering, technology, and science. You will be given a cluster "
    "of related job roles. Your task is to produce a structured JSON response describing this "
    "cluster as a career territory.\n\n"
    "Important notes on the input data:\n\n"
    "- RQF levels indicate the minimum qualification typically required for entry into a role. "
    "They do not indicate seniority. Use job titles and the entry_routes and progression prose "
    "as your primary guide to career level. Treat RQF only as a rough entry barrier indicator.\n"
    "- Salary figures are weighted averages from advertised job postings. They skew low relative "
    "to actual market rates, vary inconsistently across roles, and should be used only as a broad "
    "relative signal — not quoted as precise figures.\n\n"
    "Respond ONLY with a valid JSON object. No preamble, no explanation, no markdown fences."
)


def load_cluster(db):
    cluster = db.execute(
        "SELECT cluster_id, name FROM clusters WHERE cluster_id = ?", (CLUSTER_ID,)
    ).fetchone()
    if not cluster:
        print(f"ERROR: cluster {CLUSTER_ID} not found", file=sys.stderr)
        sys.exit(1)

    jobs = db.execute(
        """
        SELECT id, title, level, adzuna_salary_estimate, entry_routes, progression
        FROM jobs
        WHERE cluster_id = ?
        ORDER BY level, title
        """,
        (CLUSTER_ID,),
    ).fetchall()

    return cluster, jobs


def build_user_message(cluster, jobs):
    cluster_id   = cluster["cluster_id"]
    cluster_name = cluster["name"]

    # Jobs table lines
    job_lines = []
    for j in jobs:
        parts = [str(j["id"]), j["title"]]
        if j["level"] is not None:
            parts.append(f"RQF {j['level']}")
        sal = j["adzuna_salary_estimate"]
        if sal is not None and sal != 0:
            parts.append(f"~£{int(sal):,}")
        job_lines.append(" / ".join(parts))

    # Entry routes + progression per job
    detail_lines = []
    for j in jobs:
        entry       = (j["entry_routes"] or "").strip()
        progression = (j["progression"] or "").strip()
        if entry or progression:
            detail_lines.append(f"{j['title']}")
            if entry:
                detail_lines.append(f"  Entry routes: {entry}")
            if progression:
                detail_lines.append(f"  Progression: {progression}")

    # JSON schema fragment embedded in user message
    schema = (
        '{\n'
        f'  "cluster_id": "{cluster_id}",\n'
        f'  "cluster_name": "{cluster_name}",\n'
        '  "cluster_narrative": "3-4 sentence accessible description of this career territory. '
        'What kind of work, what industries, what it feels like. Written for a student audience '
        'aged 16+. Plain language, concrete and motivating.",\n'
        '  "tracks": [\n'
        '    {\n'
        '      "track_name": "Short descriptive name for this track",\n'
        '      "track_narrative": "A paragraph describing this track for a student audience. '
        'Must cover: (1) what the work actually involves day to day, (2) whether you need a '
        'degree to enter or whether apprenticeship/experience routes exist, (3) how you progress '
        'from entry level to senior roles, (4) any professional credentials or chartered status '
        'that matter (name them specifically), (5) whether there is a ceiling on non-graduate '
        'routes and what it takes to break through, (6) whether self-employment or consultancy '
        'is a realistic senior destination, (7) the industries or sectors where this work happens, '
        '(8) whether postgraduate study is expected or just advantageous, and (9) any emerging or '
        'growth areas worth knowing about. Not all points will apply to every track — include only '
        'what is genuinely relevant.",\n'
        '      "jobs": [\n'
        '        {"job_id": 123, "title": "Job title"},\n'
        '        ...ordered from entry level to most senior...\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}'
    )

    msg = (
        f"Cluster: {cluster_name}\n\n"
        "Jobs (job_id / title / RQF level / estimated salary):\n"
        + "\n".join(f"  {l}" for l in job_lines)
        + "\n\nEntry routes and progression for each job:\n"
        + "\n".join(detail_lines)
        + "\n\nProduce a JSON object with this structure:\n\n"
        + schema
        + "\n\nIdentify as many tracks as the data genuinely supports. If the cluster is a "
        "single ladder, return one track. If it has parallel tracks that do not connect, return "
        "them separately. Do not impose structure that is not there."
    )
    return msg


def write_txt(data):
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"CLUSTER {data['cluster_id']}: {data['cluster_name']}\n")
        f.write("=" * 60 + "\n\n")
        f.write("CLUSTER NARRATIVE\n")
        f.write("-" * 60 + "\n")
        f.write(data["cluster_narrative"] + "\n\n")

        for i, track in enumerate(data["tracks"], 1):
            f.write(f"TRACK {i}: {track['track_name']}\n")
            f.write("-" * 60 + "\n")
            f.write(track["track_narrative"] + "\n\n")
            f.write("Jobs (entry to senior):\n")
            for j in track["jobs"]:
                f.write(f"  [{j['job_id']}] {j['title']}\n")
            f.write("\n")


def main():
    db = sqlite3.connect(JOBS_DB)
    db.row_factory = sqlite3.Row
    client = anthropic.Anthropic()

    cluster, jobs = load_cluster(db)
    db.close()

    print(f"  Calling Haiku for cluster {CLUSTER_ID}: {cluster['name']} ({len(jobs)} jobs)...", file=sys.stderr)

    user_msg = build_user_message(cluster, jobs)

    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    print(f"  Done ({response.usage.output_tokens} tokens)", file=sys.stderr)

    # Strip markdown fences if Haiku added them despite instructions
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]          # drop opening fence line
        raw = raw.rsplit("```", 1)[0].strip() # drop closing fence

    # Parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error: {e}", file=sys.stderr)
        print(f"  Raw output:\n{raw}", file=sys.stderr)
        # Still write raw to file for inspection
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            f.write(raw)
        sys.exit(1)

    # Write JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  JSON written to: {OUTPUT_JSON}", file=sys.stderr)

    # Write plain text
    write_txt(data)
    print(f"  Text written to: {OUTPUT_TXT}", file=sys.stderr)

    # Quick summary to console
    print(f"\n  Tracks identified: {len(data['tracks'])}", file=sys.stderr)
    for i, t in enumerate(data["tracks"], 1):
        print(f"    {i}. {t['track_name']} ({len(t['jobs'])} jobs)", file=sys.stderr)


if __name__ == "__main__":
    main()
