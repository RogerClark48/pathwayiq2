"""
cluster_pathway_production2.py
Definitive cluster pathway enrichment — all 21 clusters.
Uses refined prompt (definition + examples + rules) validated in test 3.
Drops and rebuilds staging tables before running.
"""

import sqlite3
import sys
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
import anthropic

load_dotenv()

JOBS_DB    = r"C:\Dev\pathwayiq2\job_roles_asset.db"
OUTPUT_TXT = r"C:\Dev\pathwayiq2\scripts\cluster_pathway_production2_output.txt"
MODEL      = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a career guidance specialist helping students aged 16 and above understand "
    "career territories in engineering, technology, and science. You will be given a cluster "
    "of related job roles. Your task is to analyse this cluster and call the "
    "record_cluster_pathway tool with your findings.\n\n"
    "What a track is:\n\n"
    "A track represents the set of career options that a student coming from a particular "
    "educational starting point would naturally consider together. The test is not whether the "
    "jobs share a discipline, nor whether they are mechanically similar in entry requirements, "
    "but whether a student choosing one of these directions would realistically be aware of and "
    "considering the others at the same time. Jobs belong in the same track if they emerge from "
    "similar courses or training routes, exist in overlapping industries or working environments, "
    "and don't require a student to make a fundamentally different choice about their education "
    "or career direction to pursue them. Jobs belong in different tracks if a student pursuing "
    "one would typically not be considering the others — because the entry routes diverge early, "
    "the working worlds are genuinely separate, or the career commitments required are "
    "fundamentally different.\n\n"
    "Examples:\n\n"
    "Meteorologist and climate scientist belong in the same track. A student with A levels in "
    "maths and environmental science considering a degree in atmospheric science or physical "
    "geography would naturally be weighing both options at the same time. They emerge from the "
    "same degree territory, work in overlapping research and forecasting environments, and "
    "choosing one does not foreclose the other early on.\n\n"
    "Airline pilot and meteorologist do not belong in the same track. Although both work in "
    "aviation-related industries, a student pursuing a pilot licence and flying hours is in a "
    "completely different educational and career commitment from a student pursuing a science "
    "degree and postgraduate research. A student considering one would not typically be "
    "weighing the other.\n\n"
    "Rules for identifying tracks:\n\n"
    "- Identify tracks by the educational starting point and career direction a student would "
    "be choosing — not by scientific or technical discipline.\n"
    "- Only create separate tracks where the entry routes diverge early, the working worlds are "
    "genuinely separate, or the career commitments required are fundamentally different.\n"
    "- If you find yourself creating tracks with fewer than 3 jobs, try combining them with the "
    "closest related track before finalising. Only keep a small track separate if there is a "
    "genuinely strong reason why it cannot be grouped with any other track.\n"
    "- Each job must appear in exactly one track. Do not place the same job in multiple tracks.\n\n"
    "Notes on the input data:\n\n"
    "- RQF levels indicate the minimum qualification typically required for entry. They do not "
    "indicate seniority. Use job titles and the entry_routes and progression prose as your "
    "primary guide to career level.\n"
    "- Salary figures are weighted averages from advertised job postings. They skew low, vary "
    "inconsistently, and should be used only as a broad relative signal — not quoted as "
    "precise figures."
)

TOOLS = [{
    "name": "record_cluster_pathway",
    "description": "Record the career pathway analysis for a cluster",
    "input_schema": {
        "type": "object",
        "properties": {
            "cluster_narrative": {"type": "string"},
            "tracks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "track_name":      {"type": "string"},
                        "track_narrative": {"type": "string"},
                        "jobs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "job_id": {"type": "integer"},
                                    "title":  {"type": "string"},
                                },
                                "required": ["job_id", "title"],
                            },
                        },
                    },
                    "required": ["track_name", "track_narrative", "jobs"],
                },
            },
        },
        "required": ["cluster_narrative", "tracks"],
    },
}]


# ── Database ──────────────────────────────────────────────────────────────────

def rebuild_staging_tables(db):
    db.executescript("""
        DROP TABLE IF EXISTS track_jobs_staging;
        DROP TABLE IF EXISTS track_staging;
        DROP TABLE IF EXISTS cluster_pathway_staging;

        CREATE TABLE cluster_pathway_staging (
            cluster_id        TEXT PRIMARY KEY,
            cluster_name      TEXT,
            cluster_narrative TEXT,
            generated_at      TEXT
        );

        CREATE TABLE track_staging (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id       TEXT,
            track_name       TEXT,
            track_narrative  TEXT,
            track_order      INTEGER
        );

        CREATE TABLE track_jobs_staging (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            track_staging_id INTEGER,
            job_id           INTEGER,
            title            TEXT,
            job_order        INTEGER
        );
    """)
    db.commit()
    print("Staging tables dropped and rebuilt.", file=sys.stderr)


def load_cluster_jobs(db, cluster_id):
    return db.execute(
        """
        SELECT id, title, level, adzuna_salary_estimate, entry_routes, progression
        FROM jobs
        WHERE cluster_id = ?
        ORDER BY level, title
        """,
        (cluster_id,),
    ).fetchall()


def write_staging(db, cluster_id, cluster_name, result, generated_at):
    db.execute(
        """
        INSERT OR REPLACE INTO cluster_pathway_staging
            (cluster_id, cluster_name, cluster_narrative, generated_at)
        VALUES (?, ?, ?, ?)
        """,
        (cluster_id, cluster_name, result["cluster_narrative"], generated_at),
    )

    old_track_ids = [
        r[0] for r in db.execute(
            "SELECT id FROM track_staging WHERE cluster_id = ?", (cluster_id,)
        )
    ]
    if old_track_ids:
        placeholders = ",".join("?" * len(old_track_ids))
        db.execute(f"DELETE FROM track_jobs_staging WHERE track_staging_id IN ({placeholders})", old_track_ids)
        db.execute("DELETE FROM track_staging WHERE cluster_id = ?", (cluster_id,))

    for track_order, track in enumerate(result["tracks"], 1):
        cursor = db.execute(
            """
            INSERT INTO track_staging (cluster_id, track_name, track_narrative, track_order)
            VALUES (?, ?, ?, ?)
            """,
            (cluster_id, track["track_name"], track["track_narrative"], track_order),
        )
        track_id = cursor.lastrowid

        for job_order, job in enumerate(track["jobs"], 1):
            db.execute(
                """
                INSERT INTO track_jobs_staging (track_staging_id, job_id, title, job_order)
                VALUES (?, ?, ?, ?)
                """,
                (track_id, job["job_id"], job["title"], job_order),
            )

    db.commit()


# ── Prompt construction ───────────────────────────────────────────────────────

def build_user_message(cluster_name, jobs):
    job_lines = []
    for j in jobs:
        parts = [str(j["id"]), j["title"]]
        if j["level"] is not None:
            parts.append(f"RQF {j['level']}")
        sal = j["adzuna_salary_estimate"]
        if sal is not None and sal != 0:
            parts.append(f"~£{int(sal):,}")
        job_lines.append(" / ".join(parts))

    detail_lines = []
    for j in jobs:
        entry       = (j["entry_routes"] or "").strip()
        progression = (j["progression"] or "").strip()
        if entry or progression:
            detail_lines.append(j["title"])
            if entry:
                detail_lines.append(f"  Entry routes: {entry}")
            if progression:
                detail_lines.append(f"  Progression: {progression}")

    return (
        f"Cluster: {cluster_name}\n\n"
        "Jobs (job_id / title / RQF level / estimated salary):\n"
        + "\n".join(f"  {l}" for l in job_lines)
        + "\n\nEntry routes and progression for each job:\n"
        + "\n".join(detail_lines)
        + "\n\nAnalyse this cluster as a career territory and call record_cluster_pathway with:\n\n"
        "cluster_narrative: 3-4 sentences. What kind of work, what industries, what it feels "
        "like day to day. Written for a student audience aged 16+. Plain language, concrete "
        "and motivating.\n\n"
        "tracks: identify as many tracks as the data genuinely supports. For each track:\n\n"
        "  track_name: short descriptive name\n\n"
        "  track_narrative: a paragraph covering (include only what is genuinely relevant "
        "to this track):\n"
        "    - what the work actually involves day to day\n"
        "    - whether degree is required or whether apprenticeship/experience routes exist\n"
        "    - how progression works from entry to senior level\n"
        "    - professional credentials or chartered status that matter (name them specifically)\n"
        "    - whether non-graduate routes have a ceiling and what it takes to break through\n"
        "    - whether self-employment or consultancy is a realistic senior destination\n"
        "    - which industries or sectors this work happens in\n"
        "    - whether postgraduate study is expected or just advantageous\n"
        "    - any emerging or growth areas worth knowing about\n\n"
        "  jobs: ordered from entry level to most senior, as [{job_id, title}]"
    )


# ── API call ──────────────────────────────────────────────────────────────────

def call_haiku(client, cluster_name, jobs):
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        temperature=0.3,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        tool_choice={"type": "tool", "name": "record_cluster_pathway"},
        messages=[{"role": "user", "content": build_user_message(cluster_name, jobs)}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input, response.usage.output_tokens

    return None, response.usage.output_tokens


# ── Text output ───────────────────────────────────────────────────────────────

def write_txt_entry(f, cluster_id, cluster_name, result):
    f.write("=" * 60 + "\n")
    f.write(f"CLUSTER {cluster_id}: {cluster_name}\n")
    f.write("=" * 60 + "\n\n")
    f.write(result["cluster_narrative"] + "\n\n")
    for i, track in enumerate(result["tracks"], 1):
        f.write(f"TRACK {i}: {track['track_name']}\n")
        f.write("-" * 60 + "\n")
        f.write(track["track_narrative"] + "\n\n")
        f.write("Jobs (entry to senior):\n")
        for j in track["jobs"]:
            f.write(f"  [{j['job_id']}] {j['title']}\n")
        f.write("\n")
    f.write("\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    db = sqlite3.connect(JOBS_DB)
    db.row_factory = sqlite3.Row
    rebuild_staging_tables(db)

    client = anthropic.Anthropic()

    clusters = db.execute(
        "SELECT cluster_id, name FROM clusters ORDER BY cluster_id + 0, cluster_id"
    ).fetchall()

    succeeded = []
    failed    = []

    with open(OUTPUT_TXT, "w", encoding="utf-8") as txt_out:
        for cluster in clusters:
            cluster_id   = cluster["cluster_id"]
            cluster_name = cluster["name"]
            jobs         = load_cluster_jobs(db, cluster_id)

            try:
                result, tokens = call_haiku(client, cluster_name, jobs)

                if result is None:
                    raise ValueError("No tool_use block in response")

                generated_at = datetime.now(timezone.utc).isoformat()
                write_staging(db, cluster_id, cluster_name, result, generated_at)
                write_txt_entry(txt_out, cluster_id, cluster_name, result)

                track_count = len(result["tracks"])
                print(f"Cluster {cluster_id} ({cluster_name}): done — {track_count} tracks, {tokens} tokens")
                succeeded.append(cluster_id)

            except Exception as e:
                print(f"Cluster {cluster_id} ({cluster_name}): ERROR — {e}", file=sys.stderr)
                failed.append((cluster_id, cluster_name, str(e)))

            time.sleep(1)

    db.close()

    print(f"\n{'='*50}")
    print(f"Complete: {len(succeeded)}/{len(clusters)} clusters")
    if failed:
        print(f"Failed ({len(failed)}):")
        for cid, name, err in failed:
            print(f"  Cluster {cid} ({name}): {err}")
    print(f"Staging tables: {JOBS_DB}")
    print(f"Text output:    {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
