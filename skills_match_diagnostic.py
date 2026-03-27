"""
skills_match_diagnostic.py
--------------------------
Compares two cross-collection scoring approaches for all 83 GMIoT courses:

  Approach A (current):  course _overview  ->  job _overview
  Approach B (skills):   course _overview  ->  job _skills

Chunk content:
  course _overview = title + overview + what_you_will_learn  (what the course teaches)
  course _skills   = title + entry_requirements + progression (NOT used — not learning outcomes)
  job _overview    = title + overview + typical_duties
  job _skills      = title + skills_required + entry_routes + progression  (what the job needs)

Both approaches use the same course _overview vector. The difference is
which job chunk they query against — domain proximity (overview) vs
skills alignment (skills).

Read-only — no database modifications, no Voyage AI API calls.
Distance formula: score = round((1 - cosine_distance) * 100)
Both collections use hnsw:space=cosine.
"""

import sqlite3
import statistics
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_PATH = r"C:\Dev\pathwayiq\chroma_store"
GMIOT_DB    = r"C:\Dev\pathwayiq\gmiot.sqlite"
JOBS_DB     = r"C:\Dev\pathwayiq\emiot_jobs_asset.db"

TOP_N = 10

CONSISTENT_THRESHOLD = 15   # |delta| < 15% — consistent
MODERATE_THRESHOLD   = 25   # 15-25% — moderate divergence
# > 25% — large divergence

SSA_SHORT = {
    "Engineering and Manufacturing Technologies":        "Engineering",
    "Construction, Planning and the Built Environment":  "Construction",
    "Information and Communication Technology":          "ICT",
    "Health, Public Services and Care":                  "Health",
    "Arts, Media and Publishing":                        "Arts & Media",
    "Social Sciences":                                   "Social Sciences",
    "Education and Training":                            "Education",
    "Business, Administration and Law":                  "Business",
    "Retail and Commercial Enterprise":                  "Retail",
    "Leisure, Travel and Tourism":                       "Leisure",
    "Agriculture, Horticulture and Animal Care":         "Agriculture",
    "Science and Mathematics":                           "Science",
    "Languages, Literature and Culture":                 "Languages",
    "Preparation for Life and Work":                     "Prep for Work",
    "Foundation Programmes":                             "Foundation",
}

# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------
chroma      = chromadb.PersistentClient(path=CHROMA_PATH)
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
    result = collection.get(ids=[chunk_id], include=["embeddings"])
    if result["embeddings"] is not None and len(result["embeddings"]) > 0:
        return result["embeddings"][0]
    return None


def dist_to_pct(distance: float) -> int:
    """Cosine distance -> similarity percentage. Both collections use cosine space."""
    return round((1 - distance) * 100)


def divergence_tag(delta: int) -> str:
    abs_d = abs(delta)
    if abs_d < CONSISTENT_THRESHOLD:
        return "consistent"
    if abs_d < MODERATE_THRESHOLD:
        return "moderate"
    return "LARGE"


def short_ssa(ssa_label: str) -> str:
    return SSA_SHORT.get(ssa_label, ssa_label[:15])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    courses = gmiot_conn.execute(
        "SELECT course_id, course_title, ssa_label, qual_type, level "
        "FROM gmiot_courses ORDER BY course_id"
    ).fetchall()

    print("=" * 70)
    print("SKILLS MATCH DIAGNOSTIC")
    print(f"{len(courses)} courses, top-{TOP_N} career matches")
    print("A = course _overview vs job _overview  (current / domain match)")
    print("B = course _overview vs job _skills    (learning vs requirements)")
    print("=" * 70)

    # Accumulate across all courses
    all_deltas            = []   # delta for every pair that has both A and B scores
    all_a_scores          = []
    all_b_scores_matched  = []   # B scores for pairs also in A top-10
    skills_only_count     = 0    # careers in B top-10 not in A top-10
    large_neg             = []   # (delta, course_title, job_title, a_score, b_score)
    large_pos             = []
    skills_only_examples  = []   # (b_score, course_title, job_title)
    ssa_deltas            = {}   # ssa_label -> [delta, ...]
    skipped_courses       = 0

    for course in courses:
        course_id = course["course_id"]
        title     = course["course_title"]
        ssa       = course["ssa_label"] or "Unknown"
        level_str = f"Level {course['level']}" if course["level"] else ""
        ssa_tag   = short_ssa(ssa)

        vec = get_vector(courses_col, f"{course_id}_overview")
        if vec is None:
            print(f"\n[WARN] No overview vector for course {course_id} — skipping")
            skipped_courses += 1
            continue

        # --- Approach A: overview vs overview ---
        res_a = jobs_col.query(
            query_embeddings=[vec],
            n_results=TOP_N,
            where={"chunk": {"$eq": "overview"}},
            include=["metadatas", "distances"],
        )
        a_hits = [
            {
                "job_id": m["job_id"],
                "title":  m["title"],
                "source": m["source"].upper(),
                "score":  dist_to_pct(d),
            }
            for m, d in zip(res_a["metadatas"][0], res_a["distances"][0])
        ]

        # --- Approach B: overview vs skills ---
        res_b = jobs_col.query(
            query_embeddings=[vec],
            n_results=TOP_N,
            where={"chunk": {"$eq": "skills"}},
            include=["metadatas", "distances"],
        )
        # B results keyed by job_id for fast lookup
        b_by_id = {
            m["job_id"]: dist_to_pct(d)
            for m, d in zip(res_b["metadatas"][0], res_b["distances"][0])
        }
        b_ids_ordered = [m["job_id"] for m in res_b["metadatas"][0]]
        b_titles      = {m["job_id"]: m["title"] for m in res_b["metadatas"][0]}

        # --- Print per-course header ---
        print(f"\n{'=' * 70}")
        print(f"Course {course_id}: {title}  [{ssa_tag} · {level_str}]")
        print(f"{'=' * 70}")
        print(f"{'Rank':<5} {'Career':<32} {'A%':>4}  {'B%':>4}  {'Delta':>6}  Note")
        print("-" * 70)

        course_deltas = []

        for rank, hit in enumerate(a_hits, start=1):
            jid     = hit["job_id"]
            a_score = hit["score"]
            b_score = b_by_id.get(jid)  # None if not in B top-10

            all_a_scores.append(a_score)

            if b_score is not None:
                delta = b_score - a_score
                tag   = divergence_tag(delta)
                delta_str = f"{delta:+d}%"
                b_str     = f"{b_score:>3}%"
                all_deltas.append(delta)
                all_b_scores_matched.append(b_score)
                course_deltas.append(delta)

                if delta <= -MODERATE_THRESHOLD:
                    large_neg.append((delta, title, hit["title"], a_score, b_score))
                elif delta >= MODERATE_THRESHOLD:
                    large_pos.append((delta, title, hit["title"], a_score, b_score))
            else:
                delta_str = "  B<10"
                b_str     = "  --"
                tag       = ""

            print(
                f"  {rank:<3} {hit['title'][:31]:<32} {a_score:>3}%  "
                f"{b_str}  {delta_str:>6}  {tag}"
            )

        # Skills-only careers (in B top-10 but not in A top-10)
        a_ids = {h["job_id"] for h in a_hits}
        new_from_b = [(b_by_id[jid], jid) for jid in b_ids_ordered if jid not in a_ids]

        if new_from_b:
            print(f"\n  Skills-only (in B top-{TOP_N}, not in A):")
            for b_sc, jid in new_from_b:
                jt = b_titles[jid]
                print(f"       {jt[:40]:<40}  B={b_sc}%  NEW")
                skills_only_count += 1
                skills_only_examples.append((b_sc, title, jt))

        if ssa not in ssa_deltas:
            ssa_deltas[ssa] = []
        ssa_deltas[ssa].extend(course_deltas)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total_pairs = len(all_deltas)
    consistent  = sum(1 for d in all_deltas if abs(d) < CONSISTENT_THRESHOLD)
    moderate    = sum(1 for d in all_deltas if CONSISTENT_THRESHOLD <= abs(d) < MODERATE_THRESHOLD)
    large       = sum(1 for d in all_deltas if abs(d) >= MODERATE_THRESHOLD)
    large_neg_n = sum(1 for d in all_deltas if d <= -MODERATE_THRESHOLD)
    large_pos_n = sum(1 for d in all_deltas if d >= MODERATE_THRESHOLD)

    mean_a = statistics.mean(all_a_scores) if all_a_scores else 0
    med_a  = statistics.median(all_a_scores) if all_a_scores else 0
    mean_b = statistics.mean(all_b_scores_matched) if all_b_scores_matched else 0
    med_b  = statistics.median(all_b_scores_matched) if all_b_scores_matched else 0

    print("\n\n" + "=" * 70)
    print("SKILLS MATCH DIAGNOSTIC -- SUMMARY")
    print(f"{len(courses) - skipped_courses} courses, top-{TOP_N} career matches, "
          f"{total_pairs} comparable pairs")
    print("=" * 70)

    print("\nScore distributions (comparable pairs only):")
    print(f"  Approach A (overview): mean={mean_a:.1f}%  median={med_a:.0f}%  "
          f"min={min(all_a_scores)}%  max={max(all_a_scores)}%")
    print(f"  Approach B (skills):   mean={mean_b:.1f}%  median={med_b:.0f}%  "
          f"min={min(all_b_scores_matched)}%  max={max(all_b_scores_matched)}%")

    print(f"\nDivergence analysis (B% - A%, {total_pairs} pairs):")
    print(f"  Consistent (|delta| < {CONSISTENT_THRESHOLD}%):       "
          f"{consistent:>4} pairs  ({consistent/total_pairs*100:.0f}%)")
    print(f"  Moderate divergence ({CONSISTENT_THRESHOLD}-{MODERATE_THRESHOLD}%):   "
          f"{moderate:>4} pairs  ({moderate/total_pairs*100:.0f}%)")
    print(f"  Large divergence (>{MODERATE_THRESHOLD}%):           "
          f"{large:>4} pairs  ({large/total_pairs*100:.0f}%)")
    print(f"    of which skills > overview:   {large_pos_n:>3} pairs  (skills surfaces stronger connection)")
    print(f"    of which overview > skills:   {large_neg_n:>3} pairs  (overview may be spurious)")

    large_neg.sort(key=lambda x: x[0])       # most negative first
    large_pos.sort(key=lambda x: x[0], reverse=True)  # most positive first

    print(f"\nTop 10 largest negative divergences (overview > skills -- possibly spurious):")
    for delta, ct, jt, a_sc, b_sc in large_neg[:10]:
        print(f"  {ct[:30]:<30} -> {jt[:28]:<28}  A={a_sc}% B={b_sc}% d={delta:+d}%")

    print(f"\nTop 10 largest positive divergences (skills > overview -- possibly undersold):")
    for delta, ct, jt, a_sc, b_sc in large_pos[:10]:
        print(f"  {ct[:30]:<30} -> {jt[:28]:<28}  A={a_sc}% B={b_sc}% d={delta:+d}%")

    print(f"\nNew connections surfaced by skills match (B top-{TOP_N} only):")
    print(f"  {skills_only_count} careers appeared in skills top-{TOP_N} but not overview top-{TOP_N}")
    skills_only_examples.sort(key=lambda x: x[0], reverse=True)
    print(f"  Top examples:")
    for b_sc, ct, jt in skills_only_examples[:10]:
        print(f"    {ct[:30]:<30} -> {jt[:28]:<28}  B={b_sc}%")

    print(f"\nSubject area breakdown -- mean delta (B - A) by SSA:")
    for ssa_label, deltas in sorted(ssa_deltas.items()):
        if deltas:
            mean_d = statistics.mean(deltas)
            print(f"  {short_ssa(ssa_label):<20}  mean delta={mean_d:+.1f}%  "
                  f"({len(deltas)} pairs)")

    print("=" * 70)


if __name__ == "__main__":
    run()
