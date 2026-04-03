"""
mutual_match_diagnostic.py
--------------------------
Tests the hypothesis that bidirectional (mutual) cross-collection matches
are more semantically meaningful than one-directional matches.

Read-only — no changes to any database or Chroma collection.
"""

import sqlite3
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_PATH = r"C:\Dev\pathwayiq\chroma_store"
GMIOT_DB    = r"C:\Dev\pathwayiq\gmiot.sqlite"
JOBS_DB     = r"C:\Dev\pathwayiq\job_roles_asset.db"

TOP_N = 10   # career matches per course, and course matches per career

# 10 sample courses — 2 per subject area, spread for coverage
SAMPLE_COURSE_IDS = [
    # Health, Public Services and Care
    10,   # Supporting the Adult Nursing Team — T Level
    29,   # FdSc Assistant Practitioner Health and Social Care
    # Information and Communication Technology
    4,    # Digital Software Development — T Level
    27,   # HND Computing for England (Cyber Security)
    # Engineering and Manufacturing Technologies
    49,   # Automation and Control Engineering
    15,   # HNC Electrical and Electronic Engineering
    # Construction, Planning and the Built Environment
    81,   # Construction Site Supervisor Apprenticeship
    58,   # Quantity Surveying
    # Arts, Media and Publishing
    42,   # BA (Hons) Creative Practitioner (Graphics)
    74,   # DipHE Social Media Content Creation
]

# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------
chroma = chromadb.PersistentClient(path=CHROMA_PATH)
courses_col = chroma.get_collection("gmiot_courses")
jobs_col    = chroma.get_collection("gmiot_jobs")

gmiot_conn = sqlite3.connect(GMIOT_DB)
gmiot_conn.row_factory = sqlite3.Row

jobs_conn = sqlite3.connect(JOBS_DB)
jobs_conn.row_factory = sqlite3.Row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_vector(collection, chunk_id: str) -> list[float] | None:
    """Lift a stored embedding by chunk ID."""
    result = collection.get(ids=[chunk_id], include=["embeddings"])
    if result["embeddings"] is not None and len(result["embeddings"]) > 0:
        return result["embeddings"][0]
    return None


def course_info(course_id: int) -> dict | None:
    row = gmiot_conn.execute(
        "SELECT course_id, course_title, qual_type, provider, level, ssa_label "
        "FROM gmiot_courses WHERE course_id = ?",
        (course_id,)
    ).fetchone()
    return dict(row) if row else None


def job_info(job_id: str) -> dict | None:
    row = jobs_conn.execute(
        "SELECT id, title, source, salary_min, salary_max "
        "FROM jobs WHERE id = ?",
        (job_id,)
    ).fetchone()
    return dict(row) if row else None


def format_salary(salary_min: int, salary_max: int) -> str:
    if not salary_min and not salary_max:
        return "salary n/a"
    lo = f"£{salary_min // 1000}k" if salary_min else "?"
    hi = f"£{salary_max // 1000}k" if salary_max else "?"
    return f"{lo}–{hi}"


def chroma_distance_to_pct(distance: float) -> int:
    """Convert Chroma L2 distance to a 0-100 similarity score."""
    return round((1 - distance / 2) * 100)


def query_jobs_for_course(course_vector: list[float]) -> list[dict]:
    """Query gmiot_jobs for top-N careers matching this course vector."""
    result = jobs_col.query(
        query_embeddings=[course_vector],
        n_results=TOP_N,
        where={"chunk": {"$eq": "overview"}},
        include=["metadatas", "distances"],
    )
    hits = []
    for meta, dist in zip(result["metadatas"][0], result["distances"][0]):
        hits.append({
            "job_id": meta["job_id"],
            "title":  meta["title"],
            "source": meta["source"].upper(),
            "salary_min": meta.get("salary_min", 0),
            "salary_max": meta.get("salary_max", 0),
            "score": chroma_distance_to_pct(dist),
        })
    return hits


def query_courses_for_job(job_vector: list[float]) -> list[dict]:
    """Query gmiot_courses for top-N courses matching this job vector."""
    result = courses_col.query(
        query_embeddings=[job_vector],
        n_results=TOP_N,
        where={"chunk": {"$eq": "overview"}},
        include=["metadatas", "distances"],
    )
    hits = []
    for meta, dist in zip(result["metadatas"][0], result["distances"][0]):
        hits.append({
            "course_id": meta["course_id"],
            "title":     meta["course_name"],
            "score":     chroma_distance_to_pct(dist),
        })
    return hits


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------
def run():
    print("=" * 70)
    print("MUTUAL MATCH DIAGNOSTIC")
    print(f"{len(SAMPLE_COURSE_IDS)} courses sampled, top-{TOP_N} career matches, reverse-checked")
    print("=" * 70)

    all_pairs          = []      # (course_score, job_score, mutual, course_title, job_title)
    total_mutual       = 0
    total_one_dir      = 0
    mutual_course_scores   = []
    one_dir_course_scores  = []

    for course_id in SAMPLE_COURSE_IDS:
        cinfo = course_info(course_id)
        if not cinfo:
            print(f"\n[WARN] course_id {course_id} not found in DB — skipping")
            continue

        course_chunk_id = f"{course_id}_overview"
        course_vec = get_vector(courses_col, course_chunk_id)
        if course_vec is None:
            print(f"\n[WARN] No vector for {course_chunk_id} — skipping")
            continue

        level_str = f"Level {cinfo['level']}" if cinfo.get("level") else ""
        print(f"\nCourse: {cinfo['course_title']} ({cinfo['ssa_label']}, {level_str})")
        print(f"  Career matches from course side:")

        career_hits = query_jobs_for_course(course_vec)
        course_mutual_count = 0

        for hit in career_hits:
            job_chunk_id = f"{hit['job_id']}_overview"
            job_vec = get_vector(jobs_col, job_chunk_id)

            mutual_rank  = None
            mutual_score = None

            if job_vec is not None:
                reverse_hits = query_courses_for_job(job_vec)
                for rev_rank, rev_hit in enumerate(reverse_hits, start=1):
                    if str(rev_hit["course_id"]) == str(course_id):
                        mutual_rank  = rev_rank
                        mutual_score = rev_hit["score"]
                        break

            is_mutual = mutual_rank is not None
            salary_str = format_salary(hit["salary_min"], hit["salary_max"])

            if is_mutual:
                mutual_tag = f"MUTUAL (course at rank {mutual_rank}, {mutual_score}%)"
                course_mutual_count  += 1
                total_mutual         += 1
                mutual_course_scores.append(hit["score"])
            else:
                mutual_tag = f"one-directional (course not in top-{TOP_N})"
                total_one_dir        += 1
                one_dir_course_scores.append(hit["score"])

            all_pairs.append({
                "course_title": cinfo["course_title"],
                "job_title":    hit["title"],
                "course_score": hit["score"],
                "job_score":    mutual_score,
                "mutual":       is_mutual,
            })

            print(
                f"    {hit['score']:>3}%  {hit['title']:<30} [{hit['source']}]  "
                f"{salary_str:<12}  {mutual_tag}"
            )

        course_one_dir = len(career_hits) - course_mutual_count
        print(f"  Mutual: {course_mutual_count}/{len(career_hits)}  "
              f"One-directional: {course_one_dir}/{len(career_hits)}")
        print("-" * 70)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total_pairs = total_mutual + total_one_dir

    avg_mutual_score  = (sum(mutual_course_scores)  / len(mutual_course_scores)
                         if mutual_course_scores else 0)
    avg_one_dir_score = (sum(one_dir_course_scores) / len(one_dir_course_scores)
                         if one_dir_course_scores else 0)

    mutual_pct = round(total_mutual / total_pairs * 100) if total_pairs else 0

    # Top 5 mutual pairs by average of both scores
    mutual_pairs = [p for p in all_pairs if p["mutual"] and p["job_score"] is not None]
    mutual_pairs.sort(key=lambda p: (p["course_score"] + p["job_score"]) / 2, reverse=True)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Total course/career pairs tested: {total_pairs}")
    print(f"  Mutual matches:    {total_mutual} ({mutual_pct}%)")
    print(f"  One-directional:   {total_one_dir} ({100 - mutual_pct}%)")
    print()
    print("  Mutual match score distribution:")
    print(f"    Avg course->career score for mutual pairs:      {avg_mutual_score:.1f}%")
    print(f"    Avg course->career score for one-directional:  {avg_one_dir_score:.1f}%")
    print()
    print("  Top 5 strongest mutual matches overall:")
    for p in mutual_pairs[:5]:
        avg = (p["course_score"] + p["job_score"]) / 2
        print(f"    {p['course_title'][:35]:<35} <-> {p['job_title'][:30]:<30}  "
              f"({p['course_score']}% / {p['job_score']}%  avg {avg:.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run()
