"""
tag_stdcodes.py — Assign SE stdCodes to jobs appearing in connections.db.
Cosine similarity between job title embeddings and SE occupation embeddings.
Route pre-filter narrows candidates per job based on course SSA label.
No LLM calls, no Chroma — pure numpy arithmetic.
"""

import os
import sqlite3
import numpy as np
from collections import Counter

ROOT           = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SE_DB          = os.path.join(ROOT, "se_data.db")
JOBS_DB        = os.path.join(ROOT, "job_roles_asset.db")
CONNECTIONS_DB = os.path.join(ROOT, "connections.db")
GMIOT_DB       = os.path.join(ROOT, "gmiot.sqlite")

THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# SSA label → SE route name mapping
# Built from actual stored values (confirmed before writing this script)
# ---------------------------------------------------------------------------
SSA_TO_SE_ROUTE = {
    "Arts, Media and Publishing":                    "Creative and design",
    "Construction, Planning and the Built Environment": "Construction and the built environment",
    "Engineering and Manufacturing Technologies":    "Engineering and manufacturing",
    "Health, Public Services and Care":              "Health and science",
    "Information and Communication Technology":      "Digital",
    # No clean single-route equivalent — use all occupations
    "Business, Administration and Law":              None,
    "Preparation for Life and Work":                 None,
    "Social Sciences":                               None,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_columns(conn):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
    if "se_std_code" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN se_std_code TEXT")
        print("Added se_std_code column.")
    if "se_match_score" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN se_match_score REAL")
        print("Added se_match_score column.")
    conn.commit()


def normalise(matrix):
    """Row-normalise a 2D float32 array in place. Returns normalised array."""
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

    matrix = normalise(np.stack(vectors, axis=0))  # (N, 1024)
    print(f"Loaded {len(metadata)} SE occupations into memory. Matrix shape: {matrix.shape}")
    return metadata, matrix


# ---------------------------------------------------------------------------
# Step 2 — Get distinct jobs + dominant SSA label from connections table
# ---------------------------------------------------------------------------
def load_jobs_with_ssa():
    """Returns {job_id: ssa_label} — modal SSA label per job."""
    cconn = sqlite3.connect(CONNECTIONS_DB)
    cconn.execute(f"ATTACH DATABASE ? AS gmiot_db", (GMIOT_DB,))

    rows = cconn.execute("""
        SELECT c.job_id, gc.ssa_label
        FROM course_job_connections c
        JOIN gmiot_db.gmiot_courses gc ON gc.course_id = c.course_id
        WHERE gc.ssa_label IS NOT NULL
    """).fetchall()
    cconn.close()

    # Group by job_id, pick modal SSA label
    from collections import defaultdict
    job_ssa_counts = defaultdict(Counter)
    for job_id, ssa_label in rows:
        job_ssa_counts[job_id][ssa_label] += 1

    return {job_id: counter.most_common(1)[0][0]
            for job_id, counter in job_ssa_counts.items()}


# ---------------------------------------------------------------------------
# Step 3 — Load job title embeddings
# ---------------------------------------------------------------------------
def load_job_embedding(conn, job_id):
    row = conn.execute(
        "SELECT title_embedding FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    if not row or row[0] is None:
        return None
    vec = np.frombuffer(row[0], dtype=np.float32).copy()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    se_meta, se_matrix = load_se_occupations()

    # Build per-route index: route_name -> boolean mask over se_meta rows
    route_masks = {}
    all_routes = set(m[2] for m in se_meta)
    for route in all_routes:
        if route is not None:
            route_masks[route] = np.array([m[2] == route for m in se_meta])

    job_ssa = load_jobs_with_ssa()
    job_ids = sorted(job_ssa.keys())
    total   = len(job_ids)
    print(f"Distinct jobs to tag: {total}\n")

    jobs_conn = sqlite3.connect(JOBS_DB)
    ensure_columns(jobs_conn)

    # Skip already-tagged jobs
    already = set(
        r[0] for r in jobs_conn.execute(
            "SELECT id FROM jobs WHERE se_std_code IS NOT NULL"
        ).fetchall()
    )
    to_process = [jid for jid in job_ids if jid not in already]
    print(f"{len(already)} already tagged — processing {len(to_process)} jobs.\n")

    n_matched = 0
    n_null    = 0
    scores_all = []

    dist = {
        "0.90+":      0,
        "0.85-0.90":  0,
        "0.80-0.85":  0,
        "<0.80":      0,
    }

    sample_matches = []

    for i, job_id in enumerate(to_process, 1):
        ssa_label  = job_ssa[job_id]
        se_route   = SSA_TO_SE_ROUTE.get(ssa_label)  # None = use all

        job_vec = load_job_embedding(jobs_conn, job_id)
        if job_vec is None:
            jobs_conn.execute(
                "UPDATE jobs SET se_std_code = NULL, se_match_score = NULL WHERE id = ?",
                (job_id,)
            )
            n_null += 1
            continue

        # Apply route pre-filter
        if se_route is not None and se_route in route_masks:
            mask = route_masks[se_route]
            candidates = se_matrix[mask]
            cand_meta  = [m for m, keep in zip(se_meta, mask) if keep]
        else:
            candidates = se_matrix
            cand_meta  = se_meta

        if len(candidates) == 0:
            candidates = se_matrix
            cand_meta  = se_meta

        # Cosine similarity (vectors already normalised)
        scores   = np.dot(candidates, job_vec)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_std_code, best_name, _ = cand_meta[best_idx]

        scores_all.append(best_score)

        # Score distribution
        if best_score >= 0.90:
            dist["0.90+"] += 1
        elif best_score >= 0.85:
            dist["0.85-0.90"] += 1
        elif best_score >= 0.80:
            dist["0.80-0.85"] += 1
        else:
            dist["<0.80"] += 1

        if best_score >= THRESHOLD:
            std_code_out = best_std_code
            n_matched += 1
        else:
            std_code_out = None
            n_null += 1

        jobs_conn.execute(
            "UPDATE jobs SET se_std_code = ?, se_match_score = ? WHERE id = ?",
            (std_code_out, round(best_score, 4), job_id)
        )

        # Collect sample (first 10 matched)
        if std_code_out and len(sample_matches) < 10:
            title_row = jobs_conn.execute("SELECT title FROM jobs WHERE id = ?", (job_id,)).fetchone()
            job_title = title_row[0] if title_row else "?"
            sample_matches.append((job_id, job_title, best_std_code, best_name, best_score))

        if i % 50 == 0:
            jobs_conn.commit()
            print(f"[{i}/{len(to_process)}] {n_matched} matched, {n_null} null", flush=True)

    jobs_conn.commit()

    # Final summary
    print(f"\n=== stdCode Tagging Complete ===")
    print(f"Jobs processed    : {len(to_process)}")
    print(f"Matched (>=0.80)  : {n_matched}")
    print(f"Null (<0.80)      : {n_null}")
    print(f"\nScore distribution:")
    for band, count in dist.items():
        print(f"  {band:<12} : {count}")

    distinct_codes = jobs_conn.execute(
        "SELECT COUNT(DISTINCT se_std_code) FROM jobs WHERE se_std_code IS NOT NULL"
    ).fetchone()[0]
    print(f"\nDistinct stdCodes assigned : {distinct_codes}")

    print(f"\nSample matches (up to 10):")
    for job_id, job_title, std_code, occ_name, sc in sample_matches:
        print(f"  [{job_id}] {job_title[:40]:<40} -> {std_code} {occ_name[:40]} (score: {sc:.3f})")

    jobs_conn.close()
    print("\nDone.")
