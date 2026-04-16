"""
cluster_jobs.py — Exploratory cluster analysis over High-relevance job records.

Uses existing Chroma embeddings (_overview chunks). Outputs cluster_analysis.txt
to the project root. No database writes.

Run:
    C:\Dev\pathwayiq2\venv\Scripts\python.exe scripts/cluster_jobs.py
"""

import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path

import chromadb
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "job_roles_asset.db"
CHROMA_PATH  = PROJECT_ROOT / "chroma_store"
OUTPUT_PATH  = PROJECT_ROOT / "cluster_analysis.txt"

# ---------------------------------------------------------------------------
# Step 1 — Pull High-relevance jobs, deduplicate on normalized_title
# ---------------------------------------------------------------------------
print("Loading high-relevance jobs from DB...")

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Fetch all candidates, ordered so NCS rows sort before Prospects for each
# normalized_title — deduplication then keeps the first (NCS-preferred) row.
cur.execute("""
    SELECT id, title, normalized_title, source
    FROM jobs
    WHERE iot_relevant = 'high'
      AND overview IS NOT NULL
    ORDER BY
        normalized_title,
        CASE source WHEN 'ncs' THEN 0 ELSE 1 END
""")
all_rows = cur.fetchall()
conn.close()

seen_titles = set()
jobs = []
for row in all_rows:
    nt = row["normalized_title"] or row["title"]
    if nt not in seen_titles:
        seen_titles.add(nt)
        jobs.append({"id": row["id"], "title": row["title"], "source": row["source"]})

print(f"  {len(all_rows)} records -> {len(jobs)} after deduplication on normalized_title")

# ---------------------------------------------------------------------------
# Step 2 — Retrieve _overview embeddings from Chroma
# ---------------------------------------------------------------------------
print("Fetching embeddings from Chroma...")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
col = chroma_client.get_collection("gmiot_jobs")

chunk_ids = [f"{job['id']}_overview" for job in jobs]

# Retrieve in batches of 100
BATCH = 100
all_embeddings = {}
for i in range(0, len(chunk_ids), BATCH):
    batch = chunk_ids[i : i + BATCH]
    result = col.get(ids=batch, include=["embeddings"])
    for chunk_id, embedding in zip(result["ids"], result["embeddings"]):
        all_embeddings[chunk_id] = embedding

# Match back to jobs — skip any with missing chunks
matched_jobs = []
matched_embeddings = []
missing = []
for job in jobs:
    chunk_id = f"{job['id']}_overview"
    if chunk_id in all_embeddings:
        matched_jobs.append(job)
        matched_embeddings.append(all_embeddings[chunk_id])
    else:
        missing.append(job["title"])

if missing:
    print(f"  WARNING: no Chroma chunk found for {len(missing)} records — skipped:")
    for t in missing:
        print(f"    {t}")

print(f"  {len(matched_jobs)} records with embeddings ready for clustering")

# ---------------------------------------------------------------------------
# Step 3 — Build distance matrices
# ---------------------------------------------------------------------------
print("Computing distance matrices...")

matrix = np.array(matched_embeddings, dtype=np.float32)  # (n × 1024)

# Cosine distances — used for average linkage and DBSCAN
distances_condensed = pdist(matrix, metric="cosine")
distances_square    = squareform(distances_condensed)

# For Ward linkage, L2-normalise first then use Euclidean distances.
# On unit vectors, Euclidean distance is a monotone transform of cosine distance
# (euclidean² = 2 × cosine_dist), so the clusters are equivalent to cosine Ward
# but scipy's Ward criterion is mathematically valid only with Euclidean distances.
norms = np.linalg.norm(matrix, axis=1, keepdims=True)
matrix_norm = matrix / np.clip(norms, 1e-10, None)
ward_condensed = pdist(matrix_norm, metric="euclidean")

# ---------------------------------------------------------------------------
# Step 4 — Clustering
# ---------------------------------------------------------------------------
print("Running clustering...")

# Method A — Ward linkage
Z_ward      = linkage(ward_condensed, method="ward")
ward_coarse = fcluster(Z_ward, t=10, criterion="maxclust")
ward_medium = fcluster(Z_ward, t=20, criterion="maxclust")
ward_fine   = fcluster(Z_ward, t=35, criterion="maxclust")

# Method B — Average linkage
Z_avg       = linkage(distances_condensed, method="average")
avg_coarse  = fcluster(Z_avg, t=10, criterion="maxclust")
avg_medium  = fcluster(Z_avg, t=20, criterion="maxclust")
avg_fine    = fcluster(Z_avg, t=35, criterion="maxclust")

# Method C — DBSCAN
db           = DBSCAN(eps=0.10, min_samples=3, metric="precomputed")
dbscan_labels = db.fit_predict(distances_square)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
titles = [job["title"] for job in matched_jobs]

def clusters_to_groups(labels):
    """Return dict: cluster_id -> sorted list of titles. Outliers (-1) separate."""
    groups = defaultdict(list)
    for title, label in zip(titles, labels):
        groups[label].append(title)
    for label in groups:
        groups[label].sort()
    return groups


def format_hierarchical_section(label_array, granularity_name, out):
    groups = clusters_to_groups(label_array)
    n_clusters = len(groups)
    out.append(f"\n--- {granularity_name} ({n_clusters} clusters) ---\n")
    for cluster_id in sorted(groups):
        members = groups[cluster_id]
        out.append(f"Cluster {cluster_id} ({len(members)} jobs)")
        for t in members:
            out.append(f"  {t}")
        out.append("")


# ---------------------------------------------------------------------------
# Step 5 — Write output file
# ---------------------------------------------------------------------------
print(f"Writing {OUTPUT_PATH} ...")

lines = []
lines.append("IoT Job Cluster Analysis")
lines.append("========================")
lines.append(f"Total records: {len(matched_jobs)} (after deduplication)")
lines.append(f"Date: {date.today().isoformat()}")

# ---- Method A ----
lines.append("")
lines.append("=" * 52)
lines.append("METHOD A: WARD LINKAGE")
lines.append("(L2-normalised embeddings, Euclidean distances — equivalent to cosine Ward)")
lines.append("=" * 52)
format_hierarchical_section(ward_coarse, "Coarse — 10 clusters", lines)
format_hierarchical_section(ward_medium, "Medium — 20 clusters", lines)
format_hierarchical_section(ward_fine,   "Fine   — 35 clusters", lines)

# ---- Method B ----
lines.append("")
lines.append("=" * 52)
lines.append("METHOD B: AVERAGE LINKAGE")
lines.append("(cosine distances)")
lines.append("=" * 52)
format_hierarchical_section(avg_coarse, "Coarse — 10 clusters", lines)
format_hierarchical_section(avg_medium, "Medium — 20 clusters", lines)
format_hierarchical_section(avg_fine,   "Fine   — 35 clusters", lines)

# ---- Method C ----
lines.append("")
lines.append("=" * 52)
lines.append("METHOD C: DBSCAN (eps=0.10, min_samples=3)")
lines.append("(cosine distances — no preset number of clusters)")
lines.append("=" * 52)
lines.append("")

dbscan_groups = clusters_to_groups(dbscan_labels)
n_dbscan_clusters = sum(1 for k in dbscan_groups if k != -1)
n_outliers = len(dbscan_groups.get(-1, []))
lines.append(f"Clusters found: {n_dbscan_clusters}  |  Outliers: {n_outliers}")
lines.append("")

# Outliers first
if -1 in dbscan_groups:
    outlier_titles = dbscan_groups[-1]
    lines.append(f"Outliers — not assigned to any cluster ({len(outlier_titles)} jobs)")
    for t in outlier_titles:
        lines.append(f"  {t}")
    lines.append("")

for cluster_id in sorted(k for k in dbscan_groups if k != -1):
    members = dbscan_groups[cluster_id]
    lines.append(f"Cluster {cluster_id} ({len(members)} jobs)")
    for t in members:
        lines.append(f"  {t}")
    lines.append("")

OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Done. Output: {OUTPUT_PATH}")
