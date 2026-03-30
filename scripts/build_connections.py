"""
build_connections.py — Pre-compute course→job connections with Haiku gatekeeping.
Writes approved connections to connections.db.
Re-runnable — uses INSERT OR IGNORE via compound primary key.
"""

import os
import sys
import time
import json
import sqlite3
import numpy as np
import chromadb
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths — all relative to project root (one level up from scripts/)
# ---------------------------------------------------------------------------
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

CHROMA_PATH      = os.path.join(ROOT, "chroma_store")
GMIOT_DB         = os.path.join(ROOT, "gmiot.sqlite")
JOBS_DB          = os.path.join(ROOT, "job_roles_asset.db")
CONNECTIONS_DB   = os.path.join(ROOT, "connections.db")

# ---------------------------------------------------------------------------
# Thresholds — mirror api.py
# ---------------------------------------------------------------------------
CROSS_COLLECTION_MIN_DOMAIN = 75
CROSS_COLLECTION_MIN_SKILLS = 72
TOP_N_CANDIDATES            = 20   # candidates pulled from Chroma before pre-filter
HAIKU_MODEL                 = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Chroma + Anthropic clients
# ---------------------------------------------------------------------------
chroma              = chromadb.PersistentClient(path=CHROMA_PATH)
courses_col         = chroma.get_collection("gmiot_courses")
jobs_col            = chroma.get_collection("gmiot_jobs")
courses_learning_col = chroma.get_collection("gmiot_courses_learning")
jobs_skills_col     = chroma.get_collection("gmiot_jobs_skills")

ai = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from environment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def score(distance: float) -> int:
    return round((1 - distance) * 100)


def _cosine_similarity(vec_a, vec_b) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def compute_skills_score(course_id, job_id) -> int | None:
    """Skills alignment: what_you_will_learn (course) vs skills_required (job)."""
    r1 = courses_learning_col.get(ids=[f"{course_id}_learning"], include=["embeddings"])
    r2 = jobs_skills_col.get(ids=[f"{job_id}_skills_only"], include=["embeddings"])
    if r1["embeddings"] is None or len(r1["embeddings"]) == 0:
        return None
    if r2["embeddings"] is None or len(r2["embeddings"]) == 0:
        return None
    return round(_cosine_similarity(r1["embeddings"][0], r2["embeddings"][0]) * 100)


def get_all_courses():
    conn = sqlite3.connect(GMIOT_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT course_id, course_title AS title, level, qual_type, ssa_label FROM gmiot_courses"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_job_levels() -> dict:
    """Load {job_id_str: level_int_or_None} for all jobs — used for level gap filter."""
    conn = sqlite3.connect(JOBS_DB)
    rows = conn.execute("SELECT id, level FROM jobs").fetchall()
    conn.close()
    return {str(r[0]): r[1] for r in rows}


def get_job_text(job_id: str) -> str:
    conn = sqlite3.connect(JOBS_DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT title, overview, typical_duties, skills_required, entry_routes, progression "
        "FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    conn.close()
    if not row:
        return ""
    r = dict(row)
    parts = [r.get("title") or "", r.get("overview") or "", r.get("typical_duties") or "",
             r.get("skills_required") or "", r.get("entry_routes") or ""]
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Haiku gatekeeping
# ---------------------------------------------------------------------------
def haiku_gatekeep(course, candidates):
    """
    Ask Haiku to approve/reject candidates for a course.
    Fails open — approves all on any parse error.
    Returns list of approved job_id strings.
    """
    lines = []
    for c in candidates:
        lines.append(
            f"JOB_ID={c['job_id']} | {c['title']} (domain={c['domain_score']}%, skills={c['skills_score']}%)\n"
            f"{c['full_text'][:500]}"
        )

    system_prompt = (
        "You are a careers advisor. Review candidate job matches for a course and decide "
        "which are genuinely appropriate.\n\n"
        "Respond with valid JSON only — no preamble, no markdown fences. Return exactly:\n"
        '{"approved_ids": ["123", "456"], "rejected_ids": ["789"]}\n\n'
        "Approval rules:\n"
        "- APPROVE: jobs in the same subject domain as the course\n"
        "- REJECT: jobs in a clearly different domain — for example, if the course is in human "
        "healthcare (nursing, midwifery, health science), reject any veterinary, animal care, "
        "or agricultural jobs even if scores are high\n"
        "- When genuinely uncertain, approve — only reject clear domain mismatches\n"
        "- approved_ids and rejected_ids must contain the exact JOB_ID numbers from the "
        "candidates list — no other values"
    )

    user_prompt = (
        f"Course: {course['title']}\n"
        f"Qualification: {course.get('qual_type', 'Unknown')} (Level {course.get('level', 'Unknown')})\n"
        f"Subject area: {course.get('ssa_label', 'Unknown')}\n\n"
        f"Candidates are listed in order of semantic similarity — the first is the closest match.\n\n"
        f"Candidate jobs to review:\n\n"
        + "\n\n---\n\n".join(lines)
        + "\n\nReturn JSON only."
    )

    try:
        response = ai.messages.create(
            model=HAIKU_MODEL,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if Haiku adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        approved = [str(x) for x in parsed.get("approved_ids", [])]
        return approved
    except Exception as e:
        print(f"    [haiku] parse error ({e}) — failing open, approving all", flush=True)
        return [str(c["job_id"]) for c in candidates]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def init_connections_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS course_job_connections (
            course_id       INTEGER NOT NULL,
            job_id          INTEGER NOT NULL,
            semantic_score  INTEGER NOT NULL,
            skills_score    INTEGER,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (course_id, job_id)
        )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    courses = get_all_courses()
    print(f"Found {len(courses)} courses to process.\n")

    job_levels = load_job_levels()
    print(f"Loaded levels for {len(job_levels)} jobs.\n")

    conn = sqlite3.connect(CONNECTIONS_DB)
    init_connections_db(conn)

    # Full rebuild — clear existing data
    conn.execute("DELETE FROM course_job_connections")
    conn.commit()
    print("Cleared existing connections. Starting full rebuild.\n")

    zero_connection_courses = []
    total_inserted = 0

    for i, course in enumerate(courses, 1):
        cid = str(course["course_id"])

        # --- Lift course vector ---
        stored = courses_col.get(ids=[f"{cid}_overview"], include=["embeddings", "metadatas"])
        if not stored["ids"]:
            stored = courses_col.get(ids=[f"{cid}_skills"], include=["embeddings", "metadatas"])
        if not stored["ids"]:
            print(f"[{i:>2}/{len(courses)}] {course['title'][:60]} — SKIPPED (no vector)", flush=True)
            continue

        vector = stored["embeddings"][0]

        # --- Semantic search: top N candidates ---
        hits = jobs_col.query(
            query_embeddings=[vector],
            n_results=TOP_N_CANDIDATES,
            where={"chunk": {"$eq": "overview"}},
            include=["metadatas", "distances", "documents"],
        )

        matched_ids  = hits["ids"][0]
        n_candidates = len(matched_ids)

        # --- Build candidate list ---
        candidates = []
        course_level = course.get("level")  # may be None for Short Courses
        n_level_filtered = 0

        for id_, meta, dist, ov_doc in zip(
            matched_ids, hits["metadatas"][0], hits["distances"][0], hits["documents"][0]
        ):
            domain_s = score(dist)
            jid = str(meta["job_id"])

            # Level gap filter — skip if job.level > course.level + 2
            # Skip filter entirely when course level is NULL (Short Courses)
            if course_level is not None:
                job_level = job_levels.get(jid)
                if job_level is not None and job_level > course_level + 2:
                    n_level_filtered += 1
                    continue

            skills_s = compute_skills_score(cid, jid)
            full_text = get_job_text(jid)
            candidates.append({
                "job_id":       jid,
                "title":        meta.get("title", ""),
                "domain_score": domain_s,
                "skills_score": skills_s,
                "full_text":    full_text,
            })

        # --- Haiku gatekeeping ---
        approved_ids = haiku_gatekeep(course, candidates)
        n_approved = len(approved_ids)

        # --- Write to connections.db ---
        inserted = 0
        for cand in candidates:
            if cand["job_id"] in approved_ids:
                cursor = conn.execute(
                    """INSERT OR IGNORE INTO course_job_connections
                       (course_id, job_id, semantic_score, skills_score)
                       VALUES (?, ?, ?, ?)""",
                    (int(cid), int(cand["job_id"]), cand["domain_score"], cand["skills_score"])
                )
                inserted += cursor.rowcount

        conn.commit()
        total_inserted += inserted

        flag = "  *** ZERO ***" if n_approved == 0 else ""
        print(
            f"[{i:>2}/{len(courses)}] {course['title'][:55]:<55} "
            f"candidates={n_candidates} lvl-filtered={n_level_filtered} approved={n_approved}{flag}",
            flush=True
        )
        if n_approved == 0:
            zero_connection_courses.append(course["title"])

        time.sleep(0.1)  # light politeness for Anthropic API

    # --- Final summary ---
    print("\n=== Summary ===")
    total_occ = conn.execute("SELECT COUNT(DISTINCT course_id) FROM course_job_connections").fetchone()[0]
    total_pairs = conn.execute("SELECT COUNT(*) FROM course_job_connections").fetchone()[0]
    print(f"Courses with connections : {total_occ}/{len(courses)}")
    print(f"Total connection pairs  : {total_pairs}")

    if zero_connection_courses:
        print(f"\nCourses with zero approved connections ({len(zero_connection_courses)}):")
        for t in zero_connection_courses:
            print(f"  - {t}")
    else:
        print("All courses have at least one approved connection.")

    conn.close()
    print(f"\nDone. connections.db written to {CONNECTIONS_DB}")
