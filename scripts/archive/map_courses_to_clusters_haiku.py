"""
map_courses_to_clusters_haiku.py
Ask Haiku which career clusters each GMIoT course leads to.
Sample of 20 courses — independent of connections.db. Read-only.
"""

import sqlite3
import json
import os
import sys
from dotenv import load_dotenv
import anthropic

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

GMIOT_DB  = r"C:\Dev\pathwayiq2\gmiot.sqlite"
JOBS_DB   = r"C:\Dev\pathwayiq2\job_roles_asset.db"
MODEL     = "claude-haiku-4-5-20251001"

# Full run — all 83 GMIoT courses
SAMPLE_COURSE_IDS = list(range(1, 84))

SYSTEM_PROMPT = (
    "You are a careers expert advising students at an Institute of Technology in Greater Manchester. "
    "You will be given a course title and description, and a list of 21 career clusters each with "
    "their member job roles. Your task is to identify which clusters this course genuinely leads to "
    "as career destinations for graduates. Be strict — only include clusters where there is a clear, "
    "direct relationship between the course content and the cluster's job roles. "
    "For each cluster you include, also provide a certainty score from 0 to 100 indicating how "
    "confident you are in the connection (100 = unambiguous core destination, 50 = plausible but "
    "indirect, below 50 = too uncertain to include). "
    "Return a JSON array of objects only, with no explanation or reasoning. "
    'Example: [{"cluster_id": "1", "certainty": 95}, {"cluster_id": "5", "certainty": 60}]'
)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    gdb = sqlite3.connect(GMIOT_DB)
    jdb = sqlite3.connect(JOBS_DB)
    gdb.row_factory = sqlite3.Row
    jdb.row_factory = sqlite3.Row

    # Courses
    placeholders = ",".join("?" * len(SAMPLE_COURSE_IDS))
    courses = {}
    for row in gdb.execute(
        f"SELECT course_id, course_title, overview, what_you_will_learn, ssa_label, qual_type, level "
        f"FROM gmiot_courses WHERE course_id IN ({placeholders})",
        SAMPLE_COURSE_IDS,
    ):
        desc = (row["overview"] or "").strip()
        wyl  = (row["what_you_will_learn"] or "").strip()
        if wyl:
            desc = desc + "\n\nWhat you will learn:\n" + wyl
        courses[row["course_id"]] = {
            "id":       row["course_id"],
            "title":    row["course_title"],
            "ssa":      row["ssa_label"],
            "qual":     row["qual_type"],
            "level":    row["level"],
            "desc":     desc,
        }

    # Clusters with member job titles
    clusters = {}
    for row in jdb.execute("SELECT cluster_id, name FROM clusters ORDER BY cluster_id"):
        clusters[row["cluster_id"]] = {"name": row["name"], "members": []}

    for row in jdb.execute(
        "SELECT cluster_id, normalized_title FROM jobs WHERE cluster_id IS NOT NULL ORDER BY cluster_id, id"
    ):
        cid = row["cluster_id"]
        if cid in clusters:
            clusters[cid]["members"].append(row["normalized_title"])

    gdb.close()
    jdb.close()
    return courses, clusters


# ── Haiku call ────────────────────────────────────────────────────────────────

def build_cluster_context(clusters):
    lines = []
    for cid, data in sorted(clusters.items(), key=lambda x: x[0]):
        members = " | ".join(data["members"])
        lines.append(f"  {cid}: {data['name']} — {members}")
    return "\n".join(lines)


def ask_haiku(client, course, cluster_context):
    user_msg = (
        f"Course title: {course['title']}\n"
        f"Course description: {course['desc']}\n\n"
        f"Career clusters:\n{cluster_context}\n\n"
        "Which cluster_ids does this course genuinely lead to?"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    # Extract first [ to last ] to capture the full array (handles stray reasoning before/after)
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        print(f"  WARNING: unexpected Haiku output for course {course['id']}: {raw!r}", file=sys.stderr)
        return []
    try:
        items = json.loads(raw[start:end])
        # Normalise: each item must be a dict with cluster_id and certainty
        normalised = []
        for item in items:
            if isinstance(item, dict) and "cluster_id" in item:
                normalised.append({
                    "cluster_id": str(item["cluster_id"]),
                    "certainty":  int(item.get("certainty", 50)),
                })
            else:
                # Bare string fallback (shouldn't happen, but be safe)
                normalised.append({"cluster_id": str(item), "certainty": 50})
        return normalised
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"  WARNING: parse error for course {course['id']}: {e} -- raw: {raw[start:end]!r}", file=sys.stderr)
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

OUTPUT_FILE = r"C:\Dev\pathwayiq2\output\cluster_mapping_full.txt"


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    client  = anthropic.Anthropic()
    courses, clusters = load_data()
    cluster_context   = build_cluster_context(clusters)

    results = []

    for cid in SAMPLE_COURSE_IDS:
        course = courses.get(cid)
        if not course:
            print(f"WARNING: course {cid} not found", file=sys.stderr)
            continue

        print(f"  Calling Haiku for: {course['title']} (id {cid})...", file=sys.stderr)
        haiku_clusters = ask_haiku(client, course, cluster_context)
        # Sort by certainty descending so highest-confidence clusters appear first
        haiku_clusters.sort(key=lambda x: -x["certainty"])

        results.append({
            "id":       cid,
            "title":    course["title"],
            "ssa":      course["ssa"],
            "level":    course["level"],
            "clusters": haiku_clusters,
            "count":    len(haiku_clusters),
        })

    # ── Print report (stdout + file) ──────────────────────────────────────────

    out = open(OUTPUT_FILE, "w", encoding="utf-8")

    def emit(line=""):
        print(line)
        print(line, file=out)

    def cluster_label(cid):
        c = clusters.get(cid, {})
        return f"{cid}: {c.get('name', '?')}"

    def cluster_label_with_certainty(item):
        name = clusters.get(item["cluster_id"], {}).get("name", "?")
        certainty = item["certainty"]
        flag = " (?)" if certainty < 70 else ""
        return f"{item['cluster_id']}: {name} ({certainty}%){flag}"

    SEP = "=" * 100

    emit("\n" + SEP)
    emit("COURSE -> CLUSTER MAPPING REPORT (Haiku, independent of connections.db)")
    emit(SEP)

    for r in results:
        level_str = f"L{r['level']}" if r["level"] else "Short"
        emit(f"\nCourse {r['id']:3d} | {r['title']} [{level_str}]")
        emit(f"  SSA: {r['ssa']}")
        if r["clusters"]:
            labels = ", ".join(cluster_label_with_certainty(c) for c in r["clusters"])
            emit(f"  Haiku says ({r['count']}): {labels}")
        else:
            emit(f"  Haiku says: (zero clusters)")

    # ── Summary ───────────────────────────────────────────────────────────────

    emit("\n" + SEP)
    emit("SUMMARY")
    emit(SEP)

    counts = [r["count"] for r in results]
    zero_courses = [r for r in results if r["count"] == 0]

    emit("\nDistribution of cluster counts:")
    for n in range(0, max(counts) + 1):
        c = sum(1 for x in counts if x == n)
        if c:
            bar = "#" * c
            emit(f"  {n:2d} clusters: {c:2d} courses  {bar}")

    if zero_courses:
        emit(f"\nCourses with ZERO clusters ({len(zero_courses)}):")
        for r in zero_courses:
            emit(f"  [{r['id']:3d}] {r['title']}")
    else:
        emit("\nNo courses with zero clusters.")

    # Flag unexpectedly broad (5+), unexpectedly narrow (<=1 non-health), or low-certainty inclusions
    health_ssa = "Health, Public Services and Care"
    broad  = [r for r in results if r["count"] >= 5]
    narrow = [r for r in results if r["count"] <= 1 and r["ssa"] != health_ssa and r["count"] > 0]
    low_confidence = [
        (r, [c for c in r["clusters"] if c["certainty"] < 70])
        for r in results
        if any(c["certainty"] < 70 for c in r["clusters"])
    ]

    if broad:
        emit(f"\nUnexpectedly BROAD results (5+ clusters -- flag for review):")
        for r in broad:
            level_str = f"L{r['level']}" if r["level"] else "Short"
            labels = ", ".join(cluster_label_with_certainty(c) for c in r["clusters"])
            emit(f"  [{r['id']:3d}] {r['title']} [{level_str}] -> {r['count']} clusters: {labels}")

    if narrow:
        emit(f"\nUnexpectedly NARROW results (<=1 cluster, non-health -- flag for review):")
        for r in narrow:
            level_str = f"L{r['level']}" if r["level"] else "Short"
            label = cluster_label_with_certainty(r["clusters"][0]) if r["clusters"] else "(none)"
            emit(f"  [{r['id']:3d}] {r['title']} [{level_str}] -> {label}")

    if low_confidence:
        emit(f"\nLow-certainty inclusions (<70% -- borderline, inspect before using):")
        for r, low_clusters in low_confidence:
            level_str = f"L{r['level']}" if r["level"] else "Short"
            labels = ", ".join(cluster_label_with_certainty(c) for c in low_clusters)
            emit(f"  [{r['id']:3d}] {r['title']} [{level_str}]: {labels}")

    emit()
    out.close()
    print(f"\nReport written to: {OUTPUT_FILE}", file=sys.stderr)

    # ── Write to staging table ────────────────────────────────────────────────

    from datetime import datetime, timezone

    sdb = sqlite3.connect(JOBS_DB)
    sdb.execute("""
        CREATE TABLE IF NOT EXISTS course_cluster_staging (
            course_id    INTEGER NOT NULL,
            course_title TEXT NOT NULL,
            cluster_id   TEXT NOT NULL,
            cluster_name TEXT NOT NULL,
            certainty    INTEGER NOT NULL,
            accepted     INTEGER DEFAULT NULL,
            note         TEXT DEFAULT NULL,
            created_at   TEXT NOT NULL,
            PRIMARY KEY (course_id, cluster_id)
        )
    """)

    now = datetime.now(timezone.utc).isoformat()
    rows_written = 0
    for r in results:
        for c in r["clusters"]:
            sdb.execute(
                """
                INSERT OR REPLACE INTO course_cluster_staging
                    (course_id, course_title, cluster_id, cluster_name, certainty, accepted, note, created_at)
                VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
                """,
                (
                    r["id"],
                    r["title"],
                    c["cluster_id"],
                    clusters.get(c["cluster_id"], {}).get("name", "?"),
                    c["certainty"],
                    now,
                ),
            )
            rows_written += 1

    sdb.commit()
    sdb.close()
    print(f"Staging table updated: {rows_written} rows written to course_cluster_staging in {JOBS_DB}", file=sys.stderr)


if __name__ == "__main__":
    main()
