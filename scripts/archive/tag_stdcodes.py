"""
tag_stdcodes.py — Assign SE stdCodes to all jobs in job_roles_asset.db.
Stores up to 3 matches per job (score >= 0.85) in job_occupation_tags table.
No LLM calls, no Chroma — pure numpy arithmetic on pre-computed embeddings.
"""

import os
import sqlite3
import numpy as np
from collections import Counter, defaultdict

ROOT           = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SE_DB          = os.path.join(ROOT, "se_data.db")
JOBS_DB        = os.path.join(ROOT, "job_roles_asset.db")
CONNECTIONS_DB = os.path.join(ROOT, "connections.db")
GMIOT_DB       = os.path.join(ROOT, "gmiot.sqlite")

THRESHOLD = 0.85
TOP_N     = 3

# ---------------------------------------------------------------------------
# SSA label → SE route name mapping
# ---------------------------------------------------------------------------
SSA_TO_SE_ROUTE = {
    "Arts, Media and Publishing":                       "Creative and design",
    "Construction, Planning and the Built Environment": "Construction and the built environment",
    "Engineering and Manufacturing Technologies":       "Engineering and manufacturing",
    "Health, Public Services and Care":                 "Health and science",
    "Information and Communication Technology":         "Digital",
    # No clean single-route equivalent — use all occupations
    "Business, Administration and Law":                 None,
    "Preparation for Life and Work":                    None,
    "Social Sciences":                                  None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalise(matrix):
    """Row-normalise a 2D float32 array. Returns normalised array."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


# ---------------------------------------------------------------------------
# Step 1 — Load SE occupations into memory
# ---------------------------------------------------------------------------
def load_se_occupations():
    conn = sqlite3.connect(SE_DB)
    rows = conn.execute("""
        SELECT o.std_code, o.name, r.name as route_name, o.embedding
        FROM se_occupations o
        LEFT JOIN se_routes r ON o.route_id = r.route_id
        WHERE o.embedding IS NOT NULL
    """).fetchall()
    conn.close()

    metadata = []   # list of (std_code, occ_name, route_name)
    vectors  = []
    for std_code, name, route_name, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32).copy()
        metadata.append((std_code, name, route_name))
        vectors.append(vec)

    matrix = normalise(np.stack(vectors, axis=0))
    print(f"Loaded {len(metadata)} SE occupations. Matrix shape: {matrix.shape}")
    return metadata, matrix


# ---------------------------------------------------------------------------
# Step 2 — Get modal SSA label per job from connections.db (best-effort)
# Jobs not in connections.db will have no SSA label — route filter skipped.
# ---------------------------------------------------------------------------
def load_job_ssa_labels():
    """Returns {job_id: ssa_label} for jobs that appear in connections.db."""
    try:
        cconn = sqlite3.connect(CONNECTIONS_DB)
        cconn.execute("ATTACH DATABASE ? AS gmiot_db", (GMIOT_DB,))
        rows = cconn.execute("""
            SELECT c.job_id, gc.ssa_label
            FROM course_job_connections c
            JOIN gmiot_db.gmiot_courses gc ON gc.course_id = c.course_id
            WHERE gc.ssa_label IS NOT NULL
        """).fetchall()
        cconn.close()
    except Exception as e:
        print(f"Warning: could not load SSA labels ({e}) — route filtering disabled.")
        return {}

    job_ssa_counts = defaultdict(Counter)
    for job_id, ssa_label in rows:
        job_ssa_counts[job_id][ssa_label] += 1

    return {job_id: counter.most_common(1)[0][0]
            for job_id, counter in job_ssa_counts.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    se_meta, se_matrix = load_se_occupations()

    # Build per-route index
    route_masks = {}
    all_routes = set(m[2] for m in se_meta)
    for route in all_routes:
        if route is not None:
            route_masks[route] = np.array([m[2] == route for m in se_meta])

    job_ssa = load_job_ssa_labels()
    print(f"SSA labels available for {len(job_ssa)} jobs.")

    # Load all jobs with a title_embedding
    jobs_conn = sqlite3.connect(JOBS_DB)
    jobs_conn.row_factory = sqlite3.Row

    # Truncate and re-tag all jobs
    jobs_conn.execute("DELETE FROM job_occupation_tags")
    jobs_conn.commit()

    all_jobs = jobs_conn.execute(
        "SELECT id, title, title_embedding FROM jobs ORDER BY id"
    ).fetchall()
    total = len(all_jobs)
    print(f"\nTotal jobs to process: {total}\n")

    n_with_tags = 0
    n_no_match  = 0
    n_no_embed  = 0
    total_tags  = 0
    tag_dist    = {3: 0, 2: 0, 1: 0, 0: 0}
    sample_rows = []

    for i, job in enumerate(all_jobs, 1):
        job_id  = job["id"]
        blob    = job["title_embedding"]

        if blob is None:
            n_no_embed += 1
            tag_dist[0] += 1
            continue

        job_vec = np.frombuffer(blob, dtype=np.float32).copy()
        norm = np.linalg.norm(job_vec)
        if norm > 0:
            job_vec /= norm

        # Route pre-filter from SSA label if available
        ssa_label = job_ssa.get(job_id)
        se_route  = SSA_TO_SE_ROUTE.get(ssa_label) if ssa_label else None

        if se_route and se_route in route_masks:
            mask       = route_masks[se_route]
            candidates = se_matrix[mask]
            cand_meta  = [m for m, keep in zip(se_meta, mask) if keep]
        else:
            candidates = se_matrix
            cand_meta  = se_meta

        if len(candidates) == 0:
            candidates = se_matrix
            cand_meta  = se_meta

        # Cosine similarity (vectors already normalised)
        scores = np.dot(candidates, job_vec)

        # Collect all matches above threshold, sorted descending
        qualifying_idx = np.where(scores >= THRESHOLD)[0]
        if len(qualifying_idx) == 0:
            n_no_match += 1
            tag_dist[0] += 1
            if i % 25 == 0:
                print(f"[{i:>4}/{total}] {n_with_tags} tagged, {n_no_match + n_no_embed} no match  — {job['title'][:40]}", flush=True)
            continue

        # Sort qualifying by score descending, take top N
        qualifying_idx = qualifying_idx[np.argsort(scores[qualifying_idx])[::-1]]
        top = qualifying_idx[:TOP_N]

        inserted = 0
        for idx in top:
            std_code   = cand_meta[idx][0]
            occ_name   = cand_meta[idx][1]
            match_score = float(scores[idx])
            jobs_conn.execute(
                "INSERT OR REPLACE INTO job_occupation_tags (job_id, std_code, match_score) "
                "VALUES (?, ?, ?)",
                (job_id, std_code, round(match_score, 4))
            )
            inserted += 1
            if len(sample_rows) < 10:
                sample_rows.append((job_id, job["title"], std_code, occ_name, match_score))

        total_tags  += inserted
        n_with_tags += 1
        tag_dist[min(inserted, 3)] += 1

        if i % 25 == 0:
            jobs_conn.commit()
            print(f"[{i:>4}/{total}] {n_with_tags} tagged, {n_no_match + n_no_embed} no match  — {job['title'][:40]}", flush=True)

    jobs_conn.commit()

    avg_tags = total_tags / n_with_tags if n_with_tags > 0 else 0.0

    print(f"\n=== Tagging Complete ===")
    print(f"Jobs processed         : {total}")
    print(f"Jobs with >=1 tag      : {n_with_tags}")
    print(f"Jobs with 0 tags       : {n_no_match + n_no_embed}  (no embed: {n_no_embed}, below threshold: {n_no_match})")
    print(f"Total tags stored      : {total_tags}")
    print(f"Avg tags per tagged job: {avg_tags:.1f}")
    print(f"\nTag count distribution:")
    for k in [3, 2, 1, 0]:
        print(f"  {k} tag{'s' if k != 1 else ' '} : {tag_dist[k]}")

    print(f"\nSample (up to 10 rows from job_occupation_tags):")
    print(f"  {'job_id':<7} {'title':<40} {'std_code':<10} {'occupation':<40} {'score'}")
    for job_id, title, std_code, occ_name, score in sample_rows:
        print(f"  {job_id:<7} {title[:40]:<40} {std_code:<10} {occ_name[:40]:<40} {score:.3f}")

    jobs_conn.close()
    print("\nDone.")
