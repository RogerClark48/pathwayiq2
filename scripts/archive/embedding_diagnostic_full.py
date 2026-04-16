"""
embedding_diagnostic_full.py
Full embedding diagnostic — all 200 high-relevance jobs.

Generates purpose-built two-sentence descriptions via Haiku, embeds them with
Voyage AI, and compares nearest neighbours against current Chroma embeddings
across the full 200-job population.

Steps run in sequence; intermediate files are saved so individual steps can be
commented out and re-run without redoing earlier work:
  scripts/job_descriptions_200.json       — Haiku-generated descriptions
  scripts/new_embeddings_200.npy          — new Voyage AI embedding matrix
  scripts/new_embeddings_200_ids.json     — parallel job_id list

Outputs:
  scripts/embedding_diagnostic_full_output.txt    — per-job comparison
  scripts/embedding_diagnostic_full_summary.txt   — summary table (appended to above)
  scripts/embedding_diagnostic_notable.txt        — notable neighbour changes

Run from project root with venv active:
  venv/Scripts/python.exe scripts/embedding_diagnostic_full.py
"""

import json
import sqlite3
import time
from pathlib import Path

import chromadb
import numpy as np
import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
SCRIPTS_DIR    = Path(__file__).resolve().parent
DB_PATH        = PROJECT_ROOT / "job_roles_asset.db"
CHROMA_PATH    = PROJECT_ROOT / "chroma_store"

DESC_JSON      = SCRIPTS_DIR / "job_descriptions_200.json"
EMB_NPY        = SCRIPTS_DIR / "new_embeddings_200.npy"
EMB_IDS_JSON   = SCRIPTS_DIR / "new_embeddings_200_ids.json"

OUT_FULL       = SCRIPTS_DIR / "embedding_diagnostic_full_output.txt"
OUT_NOTABLE    = SCRIPTS_DIR / "embedding_diagnostic_notable.txt"

MODEL          = "claude-haiku-4-5-20251001"
TOP_N          = 10
VOYAGE_BATCH   = 20

SYSTEM_PROMPT = (
    "You are a career classification specialist. Given a job role's overview and typical duties, "
    "write exactly two sentences describing the role for the purpose of career clustering.\n\n"
    "Sentence 1: State the knowledge domain this role draws on and the professional community it "
    "belongs to — the peer group, institutions, and disciplines this person works within and "
    "identifies with professionally.\n\n"
    "Sentence 2: State the industry or sector context — where this work happens commercially or "
    "institutionally, and what organisations employ people in this role.\n\n"
    "Do not mention entry routes, qualifications, salary, career progression, or how to get into "
    "the role. Do not use the job title in your response. Write in plain declarative sentences. "
    "Two sentences only — no more, no less."
)


# ---------------------------------------------------------------------------
# Load jobs
# ---------------------------------------------------------------------------
print("Loading jobs from DB...")

db = sqlite3.connect(DB_PATH)
db.row_factory = sqlite3.Row

jobs = db.execute("""
    SELECT j.id, j.title, j.normalized_title, j.cluster_id, c.name as cluster_name,
           j.overview, j.typical_duties
    FROM jobs j
    JOIN clusters c ON c.cluster_id = j.cluster_id
    WHERE j.cluster_id IS NOT NULL
      AND j.overview IS NOT NULL
      AND j.typical_duties IS NOT NULL
    ORDER BY j.cluster_id, j.normalized_title
""").fetchall()

jobs = [dict(r) for r in jobs]
db.close()

print(f"  {len(jobs)} jobs loaded")
job_by_id = {j["id"]: j for j in jobs}
all_ids   = [j["id"] for j in jobs]


# ===========================================================================
# STEP 1 — Generate two-sentence descriptions via Haiku
# ===========================================================================
print("\n" + "=" * 60)
print("STEP 1 — Generate descriptions")
print("=" * 60)

if DESC_JSON.exists():
    descriptions = json.loads(DESC_JSON.read_text(encoding="utf-8"))
    # Keys stored as strings in JSON
    descriptions = {int(k): v for k, v in descriptions.items()}
    print(f"  Loaded {len(descriptions)} existing descriptions from {DESC_JSON.name}")
    missing_ids = [j["id"] for j in jobs if j["id"] not in descriptions]
    if missing_ids:
        print(f"  {len(missing_ids)} jobs still need descriptions — generating...")
else:
    descriptions = {}
    missing_ids = [j["id"] for j in jobs]
    print(f"  No existing file — generating all {len(missing_ids)} descriptions...")

if missing_ids:
    client = Anthropic()
    for i, jid in enumerate(missing_ids, 1):
        job = job_by_id[jid]
        user_msg = (
            f"Job title: {job['title']}\n"
            f"Cluster: {job['cluster_name']}\n\n"
            f"Overview:\n{job['overview']}\n\n"
            f"Typical duties:\n{job['typical_duties']}"
        )
        response = client.messages.create(
            model=MODEL,
            max_tokens=150,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        desc = response.content[0].text.strip()
        descriptions[jid] = desc
        print(f"  [{i:3d}/{len(missing_ids)}] {job['title']}")
        time.sleep(0.5)

    DESC_JSON.write_text(
        json.dumps({str(k): v for k, v in descriptions.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Saved to {DESC_JSON.name}")


# ===========================================================================
# STEP 2 — Embed new descriptions with Voyage AI
# ===========================================================================
print("\n" + "=" * 60)
print("STEP 2 — Embed descriptions with Voyage AI")
print("=" * 60)

if EMB_NPY.exists() and EMB_IDS_JSON.exists():
    new_matrix  = np.load(str(EMB_NPY))
    new_emb_ids = json.loads(EMB_IDS_JSON.read_text(encoding="utf-8"))
    print(f"  Loaded existing embeddings from {EMB_NPY.name} ({new_matrix.shape})")
else:
    vo = voyageai.Client()

    embed_jobs = [j for j in jobs if j["id"] in descriptions]
    texts      = [descriptions[j["id"]] for j in embed_jobs]
    new_emb_ids = [j["id"] for j in embed_jobs]

    vectors = []
    for i in range(0, len(texts), VOYAGE_BATCH):
        batch = texts[i : i + VOYAGE_BATCH]
        result = vo.embed(batch, model="voyage-3.5", input_type="document", output_dimension=1024)
        vectors.extend(result.embeddings)
        print(f"  Embedded {min(i + VOYAGE_BATCH, len(texts))}/{len(texts)}")

    new_matrix = np.array(vectors, dtype=np.float32)
    np.save(str(EMB_NPY), new_matrix)
    EMB_IDS_JSON.write_text(json.dumps(new_emb_ids), encoding="utf-8")
    print(f"  Saved to {EMB_NPY.name} {new_matrix.shape}")


# ===========================================================================
# STEP 3 — Retrieve current embeddings from Chroma
# ===========================================================================
print("\n" + "=" * 60)
print("STEP 3 — Retrieve current embeddings from Chroma")
print("=" * 60)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
col = chroma_client.get_collection("gmiot_jobs")

chunk_ids = [f"{jid}_overview" for jid in all_ids]
BATCH = 100
raw_current = {}
for i in range(0, len(chunk_ids), BATCH):
    batch = chunk_ids[i : i + BATCH]
    res = col.get(ids=batch, include=["embeddings"])
    for chunk_id, emb in zip(res["ids"], res["embeddings"]):
        jid = int(chunk_id.split("_")[0])
        raw_current[jid] = np.array(emb, dtype=np.float32)

missing_current = [j["title"] for j in jobs if j["id"] not in raw_current]
if missing_current:
    print(f"  WARNING: no current embedding for {len(missing_current)} jobs:")
    for t in missing_current:
        print(f"    {t}")
print(f"  Retrieved {len(raw_current)} current embeddings")


# ===========================================================================
# STEP 4 — Align to common job set and compute similarity matrices
# ===========================================================================
print("\n" + "=" * 60)
print("STEP 4 — Compute similarity matrices")
print("=" * 60)

# Jobs must have both current and new embeddings
new_emb_id_set = set(new_emb_ids)
valid_ids = [jid for jid in all_ids if jid in raw_current and jid in new_emb_id_set]
valid_jobs = [job_by_id[jid] for jid in valid_ids]
n = len(valid_ids)
print(f"  {n} jobs have both embedding sets")

# Build ordered matrices
new_idx   = {jid: i for i, jid in enumerate(new_emb_ids)}
curr_mat  = np.stack([raw_current[jid] for jid in valid_ids]).astype(np.float32)
new_mat   = np.stack([new_matrix[new_idx[jid]] for jid in valid_ids]).astype(np.float32)

# L2-normalise for cosine similarity via dot product
def norm_rows(m):
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-10, None)

curr_norm = norm_rows(curr_mat)
new_norm  = norm_rows(new_mat)

# Full similarity matrices — shape (n, n)
curr_sim = curr_norm @ curr_norm.T   # cosine sim
new_sim  = new_norm  @ new_norm.T

print(f"  Similarity matrices computed: {curr_sim.shape}")


# ===========================================================================
# STEP 5 — Compute top-N neighbours per job
# ===========================================================================

def top_neighbours_from_row(sim_row, self_idx, top_n):
    """Return top_n (idx, sim) pairs excluding self."""
    row = sim_row.copy()
    row[self_idx] = -1.0
    top_idx = np.argpartition(row, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
    return [(int(idx), float(row[idx])) for idx in top_idx]


# ===========================================================================
# STEP 6 — Write outputs
# ===========================================================================
print("\n" + "=" * 60)
print("STEP 5 — Writing outputs")
print("=" * 60)

full_lines    = []
notable_lines = []
summary_rows  = []

total_same_curr = 0
total_same_new  = 0

full_lines.append("Full Embedding Diagnostic — two-sentence descriptions vs current Chroma embeddings")
full_lines.append("=" * 70)
full_lines.append(f"Jobs: {n}   Top-N: {TOP_N}")
full_lines.append(f"Current: Chroma gmiot_jobs _overview chunks (title + overview + typical_duties)")
full_lines.append(f"New:     Voyage AI voyage-3.5, two-sentence domain+sector descriptions")
full_lines.append("")

for i, (jid, job) in enumerate(zip(valid_ids, valid_jobs)):
    curr_neighbours = top_neighbours_from_row(curr_sim[i], i, TOP_N)
    new_neighbours  = top_neighbours_from_row(new_sim[i],  i, TOP_N)

    curr_ids = {valid_ids[idx] for idx, _ in curr_neighbours}
    new_ids  = {valid_ids[idx] for idx, _ in new_neighbours}

    same_curr = sum(1 for idx, _ in curr_neighbours if job_by_id[valid_ids[idx]]["cluster_id"] == job["cluster_id"])
    same_new  = sum(1 for idx, _ in new_neighbours  if job_by_id[valid_ids[idx]]["cluster_id"] == job["cluster_id"])
    delta     = same_new - same_curr

    total_same_curr += same_curr
    total_same_new  += same_new

    # ── Per-job block ──
    full_lines.append("=" * 60)
    full_lines.append(f"JOB: {job['title']} (ID: {jid}) — Cluster {job['cluster_id']}: {job['cluster_name']}")
    full_lines.append("=" * 60)
    full_lines.append("")
    full_lines.append("GENERATED DESCRIPTION:")
    full_lines.append(descriptions.get(jid, "(none)"))
    full_lines.append("")

    full_lines.append(f"CURRENT EMBEDDINGS — top {TOP_N} neighbours:")
    for rank, (idx, sim) in enumerate(curr_neighbours, 1):
        nb = job_by_id[valid_ids[idx]]
        same = nb["cluster_id"] == job["cluster_id"]
        marker = "*" if same else " "
        tag    = "  [SAME]" if same else ""
        full_lines.append(f"  {rank:2d}. {marker}{nb['title']} (Cluster {nb['cluster_id']}: {nb['cluster_name']}) — sim: {sim:.3f}{tag}")
    full_lines.append("")

    full_lines.append(f"NEW EMBEDDINGS — top {TOP_N} neighbours:")
    for rank, (idx, sim) in enumerate(new_neighbours, 1):
        nb = job_by_id[valid_ids[idx]]
        same = nb["cluster_id"] == job["cluster_id"]
        marker = "*" if same else " "
        tag    = "  [SAME]" if same else ""
        full_lines.append(f"  {rank:2d}. {marker}{nb['title']} (Cluster {nb['cluster_id']}: {nb['cluster_name']}) — sim: {sim:.3f}{tag}")
    full_lines.append("")

    sign = f"+{delta}" if delta > 0 else str(delta)
    full_lines.append(f"SAME CLUSTER ({TOP_N}): current {same_curr}/{TOP_N}   new {same_new}/{TOP_N}")
    full_lines.append(f"IMPROVEMENT: {sign}")
    full_lines.append("")

    summary_rows.append((job["title"], job["cluster_id"], job["cluster_name"], same_curr, same_new, delta))

    # ── Notable changes ──
    # Dropped: in current top-N, not in new top-N
    dropped_cross = [
        valid_ids[idx] for idx in [x for x, _ in curr_neighbours]
        if valid_ids[idx] not in new_ids
           and job_by_id[valid_ids[idx]]["cluster_id"] != job["cluster_id"]
    ]
    # Gained: in new top-N, not in current top-N
    gained_same = [
        valid_ids[idx] for idx in [x for x, _ in new_neighbours]
        if valid_ids[idx] not in curr_ids
           and job_by_id[valid_ids[idx]]["cluster_id"] == job["cluster_id"]
    ]

    if dropped_cross or gained_same:
        notable_lines.append(f"JOB: {job['title']} (Cluster {job['cluster_id']}: {job['cluster_name']})")
        for nid in dropped_cross:
            nb = job_by_id[nid]
            notable_lines.append(f"  DROPPED (cross-cluster):  {nb['title']} (Cluster {nb['cluster_id']}: {nb['cluster_name']})")
        for nid in gained_same:
            nb = job_by_id[nid]
            notable_lines.append(f"  GAINED  (same-cluster):   {nb['title']} (Cluster {nb['cluster_id']}: {nb['cluster_name']})")
        notable_lines.append("")


# ── Summary table ──
max_title = max(len(r[0]) for r in summary_rows)
max_title = max(max_title, 30)
max_clust = max(len(f"{r[1]}: {r[2]}") for r in summary_rows)
max_clust = max(max_clust, 22)

full_lines.append("=" * 70)
full_lines.append("SUMMARY TABLE")
full_lines.append("=" * 70)
full_lines.append("")
hdr = f"{'Job':<{max_title}}  {'Cluster':<{max_clust}}  {'Current':>10}  {'New':>6}  {'Delta':>6}"
full_lines.append(hdr)
full_lines.append("-" * len(hdr))

for title, cid, cname, sc, sn, d in summary_rows:
    cstr  = f"{cid}: {cname}"
    dsign = f"+{d}" if d > 0 else str(d)
    full_lines.append(
        f"{title:<{max_title}}  {cstr:<{max_clust}}  {sc:>7}/{TOP_N}  {sn:>3}/{TOP_N}  {dsign:>6}"
    )

full_lines.append("-" * len(hdr))
full_lines.append(
    f"{'TOTAL':<{max_title}}  {'':<{max_clust}}  {total_same_curr:>5}/{n*TOP_N}  "
    f"{total_same_new:>1}/{n*TOP_N}  {total_same_new - total_same_curr:>+6}"
)
full_lines.append("")
full_lines.append(
    f"Within-cluster hit rate   current: {total_same_curr/(n*TOP_N)*100:.1f}%   "
    f"new: {total_same_new/(n*TOP_N)*100:.1f}%   "
    f"delta: {(total_same_new - total_same_curr)/(n*TOP_N)*100:+.1f}pp"
)
full_lines.append("")

OUT_FULL.write_text("\n".join(full_lines), encoding="utf-8")
print(f"  Written: {OUT_FULL.name}")

# ── Notable changes header ──
header_notable = [
    "Embedding Diagnostic — Notable neighbour changes",
    "=" * 70,
    f"Jobs: {n}   Top-N: {TOP_N}",
    "",
    "DROPPED (cross-cluster): neighbour present in current top-10 but absent from new top-10,",
    "  where the dropped neighbour is from a DIFFERENT cluster.",
    "  These are noisy/misleading neighbours that the new embeddings remove.",
    "",
    "GAINED (same-cluster): neighbour present in new top-10 but absent from current top-10,",
    "  where the gained neighbour is from the SAME cluster.",
    "  These are genuine improvements — correct neighbours surfaced by new embeddings.",
    "",
    f"Jobs with notable changes: {len([l for l in notable_lines if l.startswith('JOB:')])}",
    "",
]
OUT_NOTABLE.write_text(
    "\n".join(header_notable + notable_lines),
    encoding="utf-8",
)
print(f"  Written: {OUT_NOTABLE.name}")

print(f"\nResults:")
print(f"  Within-cluster hit rate — current: {total_same_curr/(n*TOP_N)*100:.1f}%  ({total_same_curr}/{n*TOP_N})")
print(f"  Within-cluster hit rate — new:     {total_same_new/(n*TOP_N)*100:.1f}%  ({total_same_new}/{n*TOP_N})")
print(f"  Delta: {(total_same_new - total_same_curr)/(n*TOP_N)*100:+.1f}pp  ({total_same_new - total_same_curr:+d} neighbours)")
