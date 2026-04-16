"""
dbscan_tune.py — Sweep DBSCAN eps values to find a sensible setting.

Loads the same embeddings as cluster_jobs.py and prints cluster/outlier counts
for a range of eps values. No file output — console only.

Run:
    C:\Dev\pathwayiq2\venv\Scripts\python.exe scripts/dbscan_tune.py
"""

import sqlite3
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "job_roles_asset.db"
CHROMA_PATH  = PROJECT_ROOT / "chroma_store"

# --- Load jobs ---
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("""
    SELECT id, title, normalized_title, source
    FROM jobs
    WHERE iot_relevant = 'high'
      AND overview IS NOT NULL
    ORDER BY normalized_title,
             CASE source WHEN 'ncs' THEN 0 ELSE 1 END
""")
all_rows = cur.fetchall()
conn.close()

seen, jobs = set(), []
for row in all_rows:
    nt = row["normalized_title"] or row["title"]
    if nt not in seen:
        seen.add(nt)
        jobs.append({"id": row["id"], "title": row["title"]})

# --- Fetch embeddings ---
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
col = chroma_client.get_collection("gmiot_jobs")

chunk_ids = [f"{job['id']}_overview" for job in jobs]
all_embeddings = {}
for i in range(0, len(chunk_ids), 100):
    batch = chunk_ids[i : i + 100]
    result = col.get(ids=batch, include=["embeddings"])
    for cid, emb in zip(result["ids"], result["embeddings"]):
        all_embeddings[cid] = emb

embeddings = [all_embeddings[f"{job['id']}_overview"]
              for job in jobs if f"{job['id']}_overview" in all_embeddings]

matrix = np.array(embeddings, dtype=np.float32)
distances_square = squareform(pdist(matrix, metric="cosine"))

print(f"Records: {len(embeddings)}\n")
print(f"{'eps':>6}  {'clusters':>8}  {'outliers':>8}  {'largest':>8}  {'smallest':>8}")
print("-" * 48)

for eps in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]:
    db = DBSCAN(eps=eps, min_samples=3, metric="precomputed")
    labels = db.fit_predict(distances_square)
    counts = Counter(labels)
    n_outliers  = counts.pop(-1, 0)
    n_clusters  = len(counts)
    largest     = max(counts.values()) if counts else 0
    smallest    = min(counts.values()) if counts else 0
    print(f"{eps:>6.2f}  {n_clusters:>8}  {n_outliers:>8}  {largest:>8}  {smallest:>8}")
