"""
cluster_elbow_analysis.py
Ward clustering elbow analysis — K=5 to K=35.

Replicates the exact embedding retrieval and pre-processing from cluster_jobs.py:
  - High-relevance jobs, deduplicated on normalized_title
  - _overview embeddings from Chroma gmiot_jobs collection
  - L2-normalised, Euclidean distances, Ward linkage

Outputs:
  scripts/cluster_elbow_analysis.txt  — table + commentary
  scripts/cluster_elbow_plot.png      — elbow curve + % improvement bars

Run from project root with venv active:
  venv/Scripts/python.exe scripts/cluster_elbow_analysis.py
"""

import sqlite3
from pathlib import Path

import chromadb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DB_PATH       = PROJECT_ROOT / "job_roles_asset.db"
CHROMA_PATH   = PROJECT_ROOT / "chroma_store"
OUTPUT_TXT    = Path(__file__).resolve().parent / "cluster_elbow_analysis.txt"
OUTPUT_PLOT   = Path(__file__).resolve().parent / "cluster_elbow_plot.png"

K_MIN, K_MAX, K_CURRENT = 5, 35, 21

# ---------------------------------------------------------------------------
# Step 1 — Pull high-relevance jobs, deduplicate on normalized_title
# (identical to cluster_jobs.py)
# ---------------------------------------------------------------------------
print("Loading high-relevance jobs from DB...")

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

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
# (identical to cluster_jobs.py)
# ---------------------------------------------------------------------------
print("Fetching embeddings from Chroma...")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
col = chroma_client.get_collection("gmiot_jobs")

chunk_ids = [f"{job['id']}_overview" for job in jobs]

BATCH = 100
all_embeddings = {}
for i in range(0, len(chunk_ids), BATCH):
    batch = chunk_ids[i : i + BATCH]
    result = col.get(ids=batch, include=["embeddings"])
    for chunk_id, embedding in zip(result["ids"], result["embeddings"]):
        all_embeddings[chunk_id] = embedding

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
    print(f"  WARNING: no Chroma chunk found for {len(missing)} records — skipped")

n_jobs = len(matched_jobs)
print(f"  {n_jobs} records with embeddings")

# ---------------------------------------------------------------------------
# Step 3 — L2-normalise and build Ward linkage
# (identical to cluster_jobs.py)
# ---------------------------------------------------------------------------
print("Building Ward linkage...")

matrix = np.array(matched_embeddings, dtype=np.float32)
norms = np.linalg.norm(matrix, axis=1, keepdims=True)
matrix_norm = matrix / np.clip(norms, 1e-10, None)

ward_condensed = pdist(matrix_norm, metric="euclidean")
Z_ward = linkage(ward_condensed, method="ward")

# ---------------------------------------------------------------------------
# Step 4 — Sweep K and compute RMSD
# ---------------------------------------------------------------------------
print(f"Sweeping K={K_MIN} to K={K_MAX}...")

def compute_rmsd(labels, embeddings):
    """Root mean square Euclidean distance from each point to its cluster centroid."""
    labels = np.array(labels)
    embeddings = np.array(embeddings)
    sq_dists = np.zeros(len(labels))
    for k in np.unique(labels):
        mask = labels == k
        centroid = embeddings[mask].mean(axis=0)
        diffs = embeddings[mask] - centroid
        sq_dists[mask] = (diffs ** 2).sum(axis=1)
    return float(np.sqrt(sq_dists.mean()))

k_values = list(range(K_MIN, K_MAX + 1))
rmsd_values = []

for k in k_values:
    labels = fcluster(Z_ward, t=k, criterion="maxclust")
    rmsd = compute_rmsd(labels, matrix_norm)
    rmsd_values.append(rmsd)
    print(f"  K={k:2d}  RMSD={rmsd:.4f}")

# ---------------------------------------------------------------------------
# Step 5 — Build table rows
# ---------------------------------------------------------------------------
rows = []
for i, (k, rmsd) in enumerate(zip(k_values, rmsd_values)):
    if i == 0:
        delta = None
        pct   = None
    else:
        delta = rmsd_values[i - 1] - rmsd   # positive = improvement
        pct   = delta / rmsd_values[i - 1] * 100
    rows.append((k, rmsd, delta, pct))

# ---------------------------------------------------------------------------
# Step 6 — Find elbow (where % improvement drops and stabilises)
# ---------------------------------------------------------------------------
pct_improvements = [r[3] for r in rows if r[3] is not None]
k_for_pct        = [r[0] for r in rows if r[3] is not None]

# Simple threshold: first K where % improvement drops below 1.5% and stays below
THRESHOLD = 1.5
elbow_k = None
for i, (k, pct) in enumerate(zip(k_for_pct, pct_improvements)):
    # Require this and the next two values to all be below threshold
    if pct < THRESHOLD:
        upcoming = pct_improvements[i : i + 3]
        if all(v < THRESHOLD for v in upcoming):
            elbow_k = k
            break

# ---------------------------------------------------------------------------
# Step 7 — Write text output
# ---------------------------------------------------------------------------
print(f"Writing {OUTPUT_TXT}...")

header = f"{'K':>4}  {'RMSD':>8}  {'Delta':>8}  {'% improve':>10}"
sep    = "-" * len(header)

lines = []
lines.append("Ward Clustering Elbow Analysis")
lines.append("=" * 60)
lines.append(f"Jobs analysed  : {n_jobs}")
lines.append(f"Embeddings     : Chroma gmiot_jobs _overview chunks")
lines.append(f"Pre-processing : L2 normalisation + Euclidean Ward")
lines.append(f"K range        : {K_MIN}–{K_MAX}")
lines.append(f"Current K      : {K_CURRENT}")
lines.append("")
lines.append(header)
lines.append(sep)
for k, rmsd, delta, pct in rows:
    if delta is None:
        lines.append(f"{k:>4}  {rmsd:>8.4f}  {'—':>8}  {'—':>10}")
    else:
        lines.append(f"{k:>4}  {rmsd:>8.4f}  {delta:>8.4f}  {pct:>9.2f}%")
lines.append("")

# Commentary
lines.append("=" * 60)
lines.append("Commentary")
lines.append("=" * 60)
lines.append("")

k21_rmsd = rmsd_values[k_values.index(K_CURRENT)]
pct_at_21 = next(r[3] for r in rows if r[0] == K_CURRENT)

if elbow_k is None:
    elbow_desc = f"no clear single elbow identified within K={K_MIN}–{K_MAX} using a {THRESHOLD}% threshold"
    elbow_vs_21 = ""
else:
    elbow_desc = f"the elbow appears at approximately K={elbow_k}, where % improvement first drops below {THRESHOLD}% and stays there"
    if elbow_k < K_CURRENT:
        elbow_vs_21 = f"K={K_CURRENT} is above the elbow — the data's natural structure is somewhat coarser than the current choice, though the additional splits may reflect deliberate editorial granularity."
    elif elbow_k == K_CURRENT:
        elbow_vs_21 = f"K={K_CURRENT} sits right at the elbow — the current cluster count aligns well with the natural structure in the data."
    else:
        elbow_vs_21 = f"K={K_CURRENT} is below the elbow — the data would support more clusters before diminishing returns set in."

# Check for secondary elbows (local peaks in % improvement after the primary elbow)
secondary = []
if elbow_k is not None:
    post_elbow_pct = [(k, p) for k, p in zip(k_for_pct, pct_improvements) if k > elbow_k]
    for k, p in post_elbow_pct:
        if p >= THRESHOLD:
            secondary.append(k)

if secondary:
    secondary_desc = f"The curve has secondary structure: % improvement spikes above {THRESHOLD}% again at K={', '.join(str(k) for k in secondary)}, suggesting a hierarchical arrangement where sub-clusters naturally emerge at higher K values."
else:
    secondary_desc = "The curve is smooth with no significant secondary elbows, suggesting the cluster structure is relatively flat rather than hierarchical."

commentary = (
    f"Using a {THRESHOLD}% marginal improvement threshold, {elbow_desc}. "
    f"{elbow_vs_21} "
    f"At K={K_CURRENT}, RMSD={k21_rmsd:.4f} with a {pct_at_21:.2f}% improvement over K={K_CURRENT-1}. "
    f"{secondary_desc}"
).strip()

lines.append(commentary)
lines.append("")

OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")

# Also print table to console
print()
print(header)
print(sep)
for k, rmsd, delta, pct in rows:
    if delta is None:
        print(f"{k:>4}  {rmsd:>8.4f}  {'—':>8}  {'—':>10}")
    else:
        print(f"{k:>4}  {rmsd:>8.4f}  {delta:>8.4f}  {pct:>9.2f}%")
print()
print("Commentary:")
print(commentary)
print()

# ---------------------------------------------------------------------------
# Step 8 — Plot
# ---------------------------------------------------------------------------
print(f"Generating plot {OUTPUT_PLOT}...")

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(10, 7),
    gridspec_kw={"height_ratios": [3, 1.5]},
    sharex=True,
)
fig.suptitle("Ward Clustering Elbow Analysis", fontsize=13, fontweight="bold")

# Main curve
ax1.plot(k_values, rmsd_values, color="#2563eb", linewidth=2, marker="o",
         markersize=4, zorder=3)
ax1.axvline(K_CURRENT, color="#dc2626", linestyle="--", linewidth=1.5,
            label=f"current (K={K_CURRENT})", zorder=2)
if elbow_k is not None:
    ax1.axvline(elbow_k, color="#16a34a", linestyle=":", linewidth=1.5,
                label=f"elbow (K={elbow_k})", zorder=2)
ax1.set_ylabel("RMSD (Euclidean from centroid)", fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title("RMSD vs K", fontsize=10, pad=4)

# % improvement bars
bar_colors = ["#94a3b8" if p < THRESHOLD else "#3b82f6" for p in pct_improvements]
ax2.bar(k_for_pct, pct_improvements, color=bar_colors, width=0.7, zorder=3)
ax2.axhline(THRESHOLD, color="#dc2626", linestyle="--", linewidth=1,
            label=f"{THRESHOLD}% threshold", zorder=2)
ax2.axvline(K_CURRENT, color="#dc2626", linestyle="--", linewidth=1.5, zorder=2)
if elbow_k is not None:
    ax2.axvline(elbow_k, color="#16a34a", linestyle=":", linewidth=1.5, zorder=2)
ax2.set_xlabel("K (number of clusters)", fontsize=10)
ax2.set_ylabel("% improvement", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_title("Marginal RMSD improvement per additional cluster", fontsize=10, pad=4)
ax2.set_xticks(k_values)
ax2.tick_params(axis="x", labelsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
plt.close()

print(f"Done.")
print(f"  Table : {OUTPUT_TXT}")
print(f"  Plot  : {OUTPUT_PLOT}")
