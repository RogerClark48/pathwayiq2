"""
course_cluster_map.py — Map GMIoT courses to Ward medium career clusters.

Parses the Ward medium (20-cluster) section from cluster_analysis.txt,
then for each course looks up its connected jobs and summarises which
career clusters they belong to.

No database writes — read-only analysis.

Run:
    C:\Dev\pathwayiq2\venv\Scripts\python.exe scripts/course_cluster_map.py
"""

import sqlite3
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
CLUSTER_FILE    = PROJECT_ROOT / "cluster_analysis.txt"
COURSES_DB      = PROJECT_ROOT / "gmiot.sqlite"
CONNECTIONS_DB  = PROJECT_ROOT / "connections.db"
JOBS_DB         = PROJECT_ROOT / "job_roles_asset.db"
OUTPUT_PATH     = PROJECT_ROOT / "course_cluster_map.txt"

# ---------------------------------------------------------------------------
# Cluster labels (Ward medium, 20 clusters)
# ---------------------------------------------------------------------------
CLUSTER_LABELS = {
    1:  "Software development",
    2:  "Games & creative digital",
    3:  "AI & data science",
    4:  "Cyber security",
    5:  "IT systems & infrastructure",
    6:  "Building surveying & compliance",
    7:  "Construction & civil engineering",
    8:  "GIS & land surveying",
    9:  "Earth & geo sciences",
    10: "Maritime & naval",
    11: "Aviation, space & weather",
    12: "Medical physics & nuclear",
    13: "Life sciences, chemistry & materials",
    14: "Energy & renewables",
    15: "Plant & maintenance engineering",
    16: "Electrical, electronics & telecoms",
    17: "Precision & metalwork manufacturing",
    18: "Mechanical, aerospace & automotive",
    19: "IT project & production management",
    20: "Chemical & process engineering",
}

# ---------------------------------------------------------------------------
# Step 1 — Parse Ward medium section from cluster_analysis.txt
# ---------------------------------------------------------------------------
print("Parsing cluster assignments from cluster_analysis.txt...")

def parse_ward_medium(path: Path) -> dict[str, int]:
    """
    Returns {display_title: cluster_id} for the Ward medium (20-cluster) section.
    Matches on section header '--- Medium' inside the Ward linkage block.
    """
    title_to_cluster = {}
    in_ward = False
    in_medium = False
    current_cluster = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")

            # Detect entry into Ward block
            if "METHOD A: WARD LINKAGE" in stripped:
                in_ward = True
                in_medium = False
                current_cluster = None
                continue

            # Detect exit from Ward block into next method
            if in_ward and "METHOD B:" in stripped:
                break

            if not in_ward:
                continue

            # Detect medium section start
            if "--- Medium" in stripped:
                in_medium = True
                current_cluster = None
                continue

            # Detect end of medium section (next granularity level)
            if in_medium and stripped.startswith("--- "):
                break

            if not in_medium:
                continue

            # Cluster header: "Cluster N (M jobs)"
            if stripped.startswith("Cluster "):
                try:
                    cluster_id = int(stripped.split()[1])
                    current_cluster = cluster_id
                except (IndexError, ValueError):
                    pass
                continue

            # Job title line: leading spaces + title
            if current_cluster is not None and stripped.startswith("  ") and stripped.strip():
                title_to_cluster[stripped.strip()] = current_cluster

    return title_to_cluster

title_to_cluster = parse_ward_medium(CLUSTER_FILE)
print(f"  {len(title_to_cluster)} job titles parsed across "
      f"{len(set(title_to_cluster.values()))} clusters")

# ---------------------------------------------------------------------------
# Step 2 — Build job_id -> cluster lookup via title + normalized_title fallback
# ---------------------------------------------------------------------------
print("Building job ID -> cluster lookup...")

conn_jobs = sqlite3.connect(JOBS_DB)
conn_jobs.row_factory = sqlite3.Row
cur_jobs = conn_jobs.cursor()
cur_jobs.execute("SELECT id, title, normalized_title FROM jobs")
all_jobs = cur_jobs.fetchall()
conn_jobs.close()

# Primary: display title -> cluster_id
# Fallback: normalized_title -> cluster_id (handles deduplicated duplicates —
# if job B was dropped in favour of job A during clustering, B shares A's
# normalized_title and should inherit A's cluster assignment)
normalized_to_cluster: dict[str, int] = {}
for display_title, cluster_id in title_to_cluster.items():
    normalized_to_cluster[display_title.lower().strip()] = cluster_id

job_id_to_cluster: dict[int, int] = {}
job_id_to_title: dict[int, str] = {}

for row in all_jobs:
    job_id_to_title[row["id"]] = row["title"] or ""
    t_norm = (row["title"] or "").lower().strip()
    nt_norm = (row["normalized_title"] or "").lower().strip()

    if t_norm in normalized_to_cluster:
        job_id_to_cluster[row["id"]] = normalized_to_cluster[t_norm]
    elif nt_norm in normalized_to_cluster:
        job_id_to_cluster[row["id"]] = normalized_to_cluster[nt_norm]

print(f"  {len(job_id_to_cluster)} of {len(all_jobs)} jobs have a cluster assignment")

# ---------------------------------------------------------------------------
# Step 3 — Load courses and connections
# ---------------------------------------------------------------------------
print("Loading courses and connections...")

conn_courses = sqlite3.connect(COURSES_DB)
conn_courses.row_factory = sqlite3.Row
cur_courses = conn_courses.cursor()
cur_courses.execute("SELECT course_id, course_title AS title, ssa_label FROM gmiot_courses ORDER BY course_title")
courses = cur_courses.fetchall()
conn_courses.close()

conn_cx = sqlite3.connect(CONNECTIONS_DB)
conn_cx.row_factory = sqlite3.Row
cur_cx = conn_cx.cursor()
cur_cx.execute("SELECT course_id, job_id FROM course_job_connections")
all_connections = cur_cx.fetchall()
conn_cx.close()

# Build course_id -> [job_ids]
course_jobs: dict[int, list[int]] = defaultdict(list)
for row in all_connections:
    course_jobs[row["course_id"]].append(row["job_id"])

print(f"  {len(courses)} courses, {len(all_connections)} connections")

# ---------------------------------------------------------------------------
# Step 4 — Map each course to clusters
# ---------------------------------------------------------------------------

def analyse_course(course_id: int) -> dict:
    job_ids = course_jobs.get(course_id, [])
    cluster_counts: Counter = Counter()
    unclassified_titles: list[str] = []

    for job_id in job_ids:
        cluster_id = job_id_to_cluster.get(job_id)
        if cluster_id:
            cluster_counts[cluster_id] += 1
        else:
            unclassified_titles.append(job_id_to_title.get(job_id, f"[id={job_id}]"))

    unclassified_titles.sort()

    ranked = cluster_counts.most_common()
    primary   = ranked[0] if ranked else None
    secondary = ranked[1] if len(ranked) >= 2 and ranked[1][1] >= 2 else None

    return {
        "job_count":    len(job_ids),
        "cluster_counts": cluster_counts,
        "ranked":       ranked,
        "primary":      primary,
        "secondary":    secondary,
        "unclassified": unclassified_titles,
        "clusters_touched": len(cluster_counts),
        "largest_cluster_count": ranked[0][1] if ranked else 0,
    }

course_results = {}
for course in courses:
    course_results[course["course_id"]] = analyse_course(course["course_id"])

# ---------------------------------------------------------------------------
# Step 5 — Write output
# ---------------------------------------------------------------------------
print(f"Writing {OUTPUT_PATH} ...")

lines = []
lines.append("GMIoT Course -> Career Cluster Mapping")
lines.append("=" * 54)
lines.append(f"Date: {date.today().isoformat()}")
lines.append(f"Courses analysed: {len(courses)}")
lines.append("Ward medium clusters: 20")
lines.append("")

SEP = "-" * 70

for course in courses:
    cid    = course["course_id"]
    result = course_results[cid]
    lines.append(SEP)
    lines.append(course["title"])
    lines.append(f"  SSA: {course['ssa_label'] or 'n/a'}")

    if result["primary"]:
        p_id, p_count = result["primary"]
        lines.append(f"  Primary cluster:   {CLUSTER_LABELS.get(p_id, f'Cluster {p_id}')} ({p_count} connections)")
    else:
        lines.append("  Primary cluster:   (none)")

    if result["secondary"]:
        s_id, s_count = result["secondary"]
        lines.append(f"  Secondary cluster: {CLUSTER_LABELS.get(s_id, f'Cluster {s_id}')} ({s_count} connections)")
    else:
        lines.append("  Secondary cluster: (none)")

    if result["unclassified"]:
        lines.append(f"  Unclassified: {', '.join(result['unclassified'])}")

    lines.append(f"  Total connections: {result['job_count']}")
    lines.append("")

# ---- Cluster coverage summary ----
lines.append("")
lines.append("=" * 54)
lines.append("CLUSTER COVERAGE SUMMARY")
lines.append("=" * 54)
lines.append("")
lines.append(f"{'Cluster':<40} {'Courses':>7}")
lines.append("-" * 50)

cluster_course_counts: Counter = Counter()
for course in courses:
    result = course_results[course["course_id"]]
    for cluster_id in result["cluster_counts"]:
        cluster_course_counts[cluster_id] += 1

for cluster_id, count in cluster_course_counts.most_common():
    label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
    lines.append(f"{label:<40} {count:>7}")

lines.append("")

# ---- Scattered courses ----
lines.append("")
lines.append("=" * 54)
lines.append("COURSES WITH SCATTERED CONNECTIONS")
lines.append("(No single cluster has more than 3 connections)")
lines.append("=" * 54)
lines.append("")

scattered = [
    (course, course_results[course["course_id"]])
    for course in courses
    if course_results[course["course_id"]]["largest_cluster_count"] <= 3
    and course_results[course["course_id"]]["job_count"] > 0
]

if scattered:
    for course, result in sorted(scattered, key=lambda x: -x[1]["clusters_touched"]):
        lines.append(
            f"{course['title']} -- "
            f"{result['clusters_touched']} clusters touched, "
            f"largest has {result['largest_cluster_count']} connections"
        )
else:
    lines.append("(none)")

OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Done. Output: {OUTPUT_PATH}")
