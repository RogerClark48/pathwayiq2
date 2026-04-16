"""
cluster_elbow_analysis_new.py
Ward clustering elbow analysis on new two-sentence description embeddings.

Loads new_embeddings_200.npy / new_embeddings_200_ids.json (produced by
embedding_diagnostic_full.py), looks up cluster assignments from the DB,
then runs the same Ward + RMSD sweep as cluster_elbow_analysis.py.

Outputs:
  scripts/cluster_elbow_analysis_new.txt
  scripts/cluster_elbow_analysis_new.png

Run from project root with venv active:
  venv/Scripts/python.exe scripts/cluster_elbow_analysis_new.py
"""

import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR  = Path(__file__).resolve().parent
DB_PATH      = PROJECT_ROOT / "job_roles_asset.db"

EMB_NPY      = SCRIPTS_DIR / "new_embeddings_200.npy"
EMB_IDS_JSON = SCRIPTS_DIR / "new_embeddings_200_ids.json"
OUTPUT_TXT   = SCRIPTS_DIR / "cluster_elbow_analysis_new.txt"
OUTPUT_PLOT  = SCRIPTS_DIR / "cluster_elbow_analysis_new.png"

K_MIN, K_MAX, K_CURRENT = 5, 35, 21

# ---------------------------------------------------------------------------
# Load embeddings and cluster assignments
# ---------------------------------------------------------------------------
print("Loading new embeddings...")
matrix_raw = np.load(str(EMB_NPY)).astype(np.float32)
job_ids    = json.loads(EMB_IDS_JSON.read_text(encoding="utf-8"))
print(f"  {matrix_raw.shape[0]} embeddings loaded")

db = sqlite3.connect(DB_PATH)
db.row_factory = sqlite3.Row
rows = db.execute(
    f"SELECT id, cluster_id FROM jobs WHERE id IN ({','.join('?'*len(job_ids))})",
    job_ids,
).fetchall()
db.close()

cluster_map = {r["id"]: r["cluster_id"] for r in rows}
cluster_ids = [cluster_map.get(jid) for jid in job_ids]

# Drop any jobs without a cluster assignment
valid_mask = [cid is not None for cid in cluster_ids]
matrix_raw  = matrix_raw[valid_mask]
cluster_ids = [c for c, m in zip(cluster_ids, valid_mask) if m]
n_jobs = len(cluster_ids)
print(f"  {n_jobs} jobs with cluster assignments")

# ---------------------------------------------------------------------------
# L2-normalise → Ward linkage (same pre-processing as cluster_jobs.py)
# ---------------------------------------------------------------------------
print("Building Ward linkage...")
norms        = np.linalg.norm(matrix_raw, axis=1, keepdims=True)
matrix_norm  = matrix_raw / np.clip(norms, 1e-10, None)
ward_condensed = pdist(matrix_norm, metric="euclidean")
Z_ward         = linkage(ward_condensed, method="ward")

# ---------------------------------------------------------------------------
# RMSD helper
# ---------------------------------------------------------------------------
def compute_rmsd(labels, embeddings):
    labels     = np.array(labels)
    embeddings = np.array(embeddings)
    sq_dists   = np.zeros(len(labels))
    for k in np.unique(labels):
        mask     = labels == k
        centroid = embeddings[mask].mean(axis=0)
        diffs    = embeddings[mask] - centroid
        sq_dists[mask] = (diffs ** 2).sum(axis=1)
    return float(np.sqrt(sq_dists.mean()))

# ---------------------------------------------------------------------------
# Sweep K
# ---------------------------------------------------------------------------
print(f"Sweeping K={K_MIN} to K={K_MAX}...")

k_values   = list(range(K_MIN, K_MAX + 1))
rmsd_values = []

for k in k_values:
    labels = fcluster(Z_ward, t=k, criterion="maxclust")
    rmsd   = compute_rmsd(labels, matrix_norm)
    rmsd_values.append(rmsd)
    print(f"  K={k:2d}  RMSD={rmsd:.4f}")

# ---------------------------------------------------------------------------
# Build table rows
# ---------------------------------------------------------------------------
rows_out = []
for i, (k, rmsd) in enumerate(zip(k_values, rmsd_values)):
    if i == 0:
        delta, pct = None, None
    else:
        delta = rmsd_values[i - 1] - rmsd
        pct   = delta / rmsd_values[i - 1] * 100
    rows_out.append((k, rmsd, delta, pct))

# ---------------------------------------------------------------------------
# Find elbow
# ---------------------------------------------------------------------------
THRESHOLD       = 1.5
pct_improvements = [r[3] for r in rows_out if r[3] is not None]
k_for_pct        = [r[0] for r in rows_out if r[3] is not None]

elbow_k = None
for i, (k, pct) in enumerate(zip(k_for_pct, pct_improvements)):
    upcoming = pct_improvements[i : i + 3]
    if pct < THRESHOLD and all(v < THRESHOLD for v in upcoming):
        elbow_k = k
        break

# ---------------------------------------------------------------------------
# Write text output
# ---------------------------------------------------------------------------
print(f"Writing {OUTPUT_TXT.name}...")

header = f"{'K':>4}  {'RMSD':>8}  {'Delta':>8}  {'% improve':>10}"
sep    = "-" * len(header)

lines = []
lines.append("Ward Clustering Elbow Analysis — NEW two-sentence description embeddings")
lines.append("=" * 70)
lines.append(f"Jobs analysed  : {n_jobs}")
lines.append(f"Embeddings     : Voyage AI voyage-3.5, two-sentence domain+sector descriptions")
lines.append(f"Pre-processing : L2 normalisation + Euclidean Ward (same as original analysis)")
lines.append(f"K range        : {K_MIN}–{K_MAX}")
lines.append(f"Current K      : {K_CURRENT}")
lines.append("")
lines.append(header)
lines.append(sep)
for k, rmsd, delta, pct in rows_out:
    if delta is None:
        lines.append(f"{k:>4}  {rmsd:>8.4f}  {'—':>8}  {'—':>10}")
    else:
        lines.append(f"{k:>4}  {rmsd:>8.4f}  {delta:>8.4f}  {pct:>9.2f}%")
lines.append("")

lines.append("=" * 70)
lines.append("Commentary")
lines.append("=" * 70)
lines.append("")

k21_rmsd   = rmsd_values[k_values.index(K_CURRENT)]
pct_at_21  = next(r[3] for r in rows_out if r[0] == K_CURRENT)

if elbow_k is None:
    elbow_desc   = f"no clear single elbow identified within K={K_MIN}–{K_MAX} using a {THRESHOLD}% threshold"
    elbow_vs_21  = ""
else:
    elbow_desc = f"the elbow appears at approximately K={elbow_k}"
    if elbow_k < K_CURRENT:
        elbow_vs_21 = f"K={K_CURRENT} is above the elbow — the natural structure is somewhat coarser."
    elif elbow_k == K_CURRENT:
        elbow_vs_21 = f"K={K_CURRENT} sits right at the elbow — a natural match."
    else:
        elbow_vs_21 = f"K={K_CURRENT} is below the elbow — the data supports more clusters."

secondary = []
if elbow_k is not None:
    for k, p in zip(k_for_pct, pct_improvements):
        if k > elbow_k and p >= THRESHOLD:
            secondary.append(k)

secondary_desc = (
    f"Secondary spikes above {THRESHOLD}% at K={', '.join(str(k) for k in secondary)}, suggesting hierarchical sub-structure."
    if secondary else
    "The curve is smooth with no secondary elbows."
)

commentary = (
    f"Using a {THRESHOLD}% marginal improvement threshold, {elbow_desc}. "
    f"{elbow_vs_21} "
    f"At K={K_CURRENT}, RMSD={k21_rmsd:.4f} with a {pct_at_21:.2f}% improvement over K={K_CURRENT-1}. "
    f"{secondary_desc}"
).strip()

lines.append(commentary)
lines.append("")
OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")

print()
print(header)
print(sep)
for k, rmsd, delta, pct in rows_out:
    if delta is None:
        print(f"{k:>4}  {rmsd:>8.4f}  {'—':>8}  {'—':>10}")
    else:
        print(f"{k:>4}  {rmsd:>8.4f}  {delta:>8.4f}  {pct:>9.2f}%")
print()
print("Commentary:")
print(commentary)
print()

# ---------------------------------------------------------------------------
# Plot — two panels: RMSD curve + % improvement bars
# ---------------------------------------------------------------------------
print(f"Generating {OUTPUT_PLOT.name}...")

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 7),
    gridspec_kw={"height_ratios": [3, 1.5]},
    sharex=True,
)
fig.suptitle(
    "Ward Clustering Elbow Analysis — New Description Embeddings",
    fontsize=13, fontweight="bold",
)

ax1.plot(k_values, rmsd_values, color="#2563eb", linewidth=2, marker="o",
         markersize=4, zorder=3, label="new embeddings")
ax1.axvline(K_CURRENT, color="#dc2626", linestyle="--", linewidth=1.5,
            label=f"current (K={K_CURRENT})", zorder=2)
if elbow_k is not None:
    ax1.axvline(elbow_k, color="#16a34a", linestyle=":", linewidth=1.5,
                label=f"elbow (K={elbow_k})", zorder=2)
ax1.set_ylabel("RMSD (Euclidean from centroid)", fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title("RMSD vs K", fontsize=10, pad=4)

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
plt.savefig(str(OUTPUT_PLOT), dpi=150, bbox_inches="tight")
plt.close()

print(f"Done.")
print(f"  Table : {OUTPUT_TXT}")
print(f"  Plot  : {OUTPUT_PLOT}")
