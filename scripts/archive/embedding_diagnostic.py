"""
embedding_diagnostic.py
Test whether purpose-built two-sentence job descriptions produce better
nearest-neighbour relationships than the current Chroma embeddings.

Diagnostic only — no writes to the production Chroma store.

Outputs:
  scripts/embedding_diagnostic_output.txt

Run from project root with venv active:
  venv/Scripts/python.exe scripts/embedding_diagnostic.py
"""

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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "job_roles_asset.db"
CHROMA_PATH  = PROJECT_ROOT / "chroma_store"
OUTPUT_TXT   = Path(__file__).resolve().parent / "embedding_diagnostic_output.txt"

MODEL = "claude-haiku-4-5-20251001"

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
# Step 1 — Sample selection: 2 jobs per cluster, alphabetically first and last
# ---------------------------------------------------------------------------
print("Selecting sample jobs...")

db = sqlite3.connect(DB_PATH)
db.row_factory = sqlite3.Row

rows = db.execute("""
    SELECT j.id, j.title, j.normalized_title, j.cluster_id, c.name as cluster_name,
           j.overview, j.typical_duties
    FROM jobs j
    JOIN clusters c ON c.cluster_id = j.cluster_id
    WHERE j.overview IS NOT NULL AND j.typical_duties IS NOT NULL
    ORDER BY j.cluster_id, j.normalized_title
""").fetchall()

db.close()

# Group by cluster_id; pick first and last by normalized_title
from collections import defaultdict
cluster_groups = defaultdict(list)
for r in rows:
    cluster_groups[r["cluster_id"]].append(r)

sample = []
for cid in sorted(cluster_groups, key=lambda x: (len(x), x)):
    group = cluster_groups[cid]
    if len(group) == 1:
        sample.append(dict(group[0]))
    else:
        first = dict(group[0])
        last  = dict(group[-1])
        sample.append(first)
        if last["id"] != first["id"]:
            sample.append(last)

print(f"  Sample: {len(sample)} jobs across {len(cluster_groups)} clusters")
for job in sample:
    print(f"    [{job['cluster_id']:>5}] {job['title']}")


# ---------------------------------------------------------------------------
# Step 2 — Generate two-sentence descriptions via Haiku
# ---------------------------------------------------------------------------
print("\nGenerating descriptions via Haiku...")

client = anthropic.Anthropic() if False else Anthropic()

descriptions = {}
for i, job in enumerate(sample, 1):
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
    descriptions[job["id"]] = desc
    print(f"  [{i:2d}/{len(sample)}] {job['title']}: {desc[:80]}...")
    time.sleep(0.5)


# ---------------------------------------------------------------------------
# Step 3 — Embed generated descriptions with Voyage AI
# ---------------------------------------------------------------------------
print("\nEmbedding generated descriptions with Voyage AI...")

vo = voyageai.Client()

texts = [descriptions[job["id"]] for job in sample]
result = vo.embed(texts, model="voyage-3.5", input_type="document", output_dimension=1024)

new_embeddings = {}
for job, vec in zip(sample, result.embeddings):
    new_embeddings[job["id"]] = np.array(vec, dtype=np.float32)

print(f"  Embedded {len(new_embeddings)} descriptions")


# ---------------------------------------------------------------------------
# Step 4 — Retrieve current embeddings from Chroma
# ---------------------------------------------------------------------------
print("Retrieving current embeddings from Chroma...")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
col = chroma_client.get_collection("gmiot_jobs")

chunk_ids = [f"{job['id']}_overview" for job in sample]
BATCH = 100
raw = {}
for i in range(0, len(chunk_ids), BATCH):
    batch = chunk_ids[i : i + BATCH]
    result_chroma = col.get(ids=batch, include=["embeddings"])
    for chunk_id, emb in zip(result_chroma["ids"], result_chroma["embeddings"]):
        job_id = int(chunk_id.split("_")[0])
        raw[job_id] = np.array(emb, dtype=np.float32)

current_embeddings = {}
missing = []
for job in sample:
    if job["id"] in raw:
        current_embeddings[job["id"]] = raw[job["id"]]
    else:
        missing.append(job["title"])

if missing:
    print(f"  WARNING: no current embedding for: {', '.join(missing)}")

print(f"  Retrieved {len(current_embeddings)} current embeddings")


# ---------------------------------------------------------------------------
# Step 5 — Cosine similarity helpers
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def top_neighbours(job_id, emb_dict, sample, n=5):
    """Return top-n neighbours for job_id, excluding itself, as list of (job, sim)."""
    scores = []
    vec = emb_dict[job_id]
    for other in sample:
        if other["id"] == job_id:
            continue
        if other["id"] not in emb_dict:
            continue
        sim = cosine_sim(vec, emb_dict[other["id"]])
        scores.append((other, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]


# Build a lookup: id -> job dict
job_by_id = {job["id"]: job for job in sample}

# ---------------------------------------------------------------------------
# Step 6 — Write comparison output
# ---------------------------------------------------------------------------
print(f"\nWriting {OUTPUT_TXT}...")

lines = []
lines.append("Embedding Diagnostic — Two-sentence descriptions vs current Chroma embeddings")
lines.append("=" * 70)
lines.append(f"Sample: {len(sample)} jobs, {len(cluster_groups)} clusters")
lines.append(f"Current embeddings: Chroma gmiot_jobs _overview chunks")
lines.append(f"New embeddings: Voyage AI voyage-3.5 (document), two-sentence descriptions")
lines.append("")

summary_rows = []   # (title, cluster_id, cluster_name, same_current, same_new)

valid_sample = [j for j in sample if j["id"] in current_embeddings and j["id"] in new_embeddings]

for job in valid_sample:
    jid  = job["id"]
    desc = descriptions.get(jid, "(no description generated)")

    curr_neighbours = top_neighbours(jid, current_embeddings, valid_sample)
    new_neighbours  = top_neighbours(jid, new_embeddings,     valid_sample)

    same_curr = sum(1 for n, _ in curr_neighbours if n["cluster_id"] == job["cluster_id"])
    same_new  = sum(1 for n, _ in new_neighbours  if n["cluster_id"] == job["cluster_id"])

    lines.append("=" * 60)
    lines.append(f"JOB: {job['title']} (ID: {jid}) — Cluster {job['cluster_id']}: {job['cluster_name']}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("GENERATED DESCRIPTION:")
    lines.append(desc)
    lines.append("")

    lines.append("CURRENT EMBEDDINGS — top 5 neighbours:")
    for rank, (n, sim) in enumerate(curr_neighbours, 1):
        marker = "*" if n["cluster_id"] == job["cluster_id"] else " "
        lines.append(f"  {rank}. {marker}{n['title']} (Cluster {n['cluster_id']}: {n['cluster_name']}) — similarity: {sim:.3f}")
    lines.append("")

    lines.append("NEW EMBEDDINGS — top 5 neighbours:")
    for rank, (n, sim) in enumerate(new_neighbours, 1):
        marker = "*" if n["cluster_id"] == job["cluster_id"] else " "
        lines.append(f"  {rank}. {marker}{n['title']} (Cluster {n['cluster_id']}: {n['cluster_name']}) — similarity: {sim:.3f}")
    lines.append("")

    lines.append(f"SAME CLUSTER — current: {same_curr}/5  new: {same_new}/5")
    lines.append(f"CROSS-CLUSTER — current: {5 - same_curr}/5  new: {5 - same_new}/5")
    lines.append("")

    summary_rows.append((job["title"], job["cluster_id"], job["cluster_name"], same_curr, same_new))


# ---------------------------------------------------------------------------
# Step 7 — Summary table
# ---------------------------------------------------------------------------
lines.append("=" * 70)
lines.append("SUMMARY TABLE")
lines.append("=" * 70)
lines.append("")

col1 = max(len(r[0]) for r in summary_rows) + 2
col2 = max(len(f"{r[1]}: {r[2]}") for r in summary_rows) + 2
col1 = max(col1, 30)
col2 = max(col2, 20)

hdr = f"{'Job':<{col1}} {'Cluster':<{col2}} {'Current':>9} {'New':>6}"
lines.append(hdr)
lines.append(f"{'Same-cluster neighbours (out of 5)':>{col1 + col2 + 4}}")
lines.append("-" * (col1 + col2 + 18))

total_curr = 0
total_new  = 0
for title, cid, cname, sc, sn in summary_rows:
    cluster_str = f"{cid}: {cname}"
    lines.append(f"{title:<{col1}} {cluster_str:<{col2}} {sc:>7}/5  {sn:>3}/5")
    total_curr += sc
    total_new  += sn

n = len(summary_rows)
lines.append("-" * (col1 + col2 + 18))
lines.append(f"{'TOTAL':<{col1}} {'':<{col2}} {total_curr:>5}/{n*5}  {total_new:>1}/{n*5}")
lines.append("")
lines.append(
    f"Within-cluster hit rate — current: {total_curr/(n*5)*100:.1f}%   "
    f"new: {total_new/(n*5)*100:.1f}%"
)
lines.append("")

OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")

# Console summary
print(f"\nResults:")
print(f"  Within-cluster hit rate — current: {total_curr/(n*5)*100:.1f}%  ({total_curr}/{n*5})")
print(f"  Within-cluster hit rate — new:     {total_new/(n*5)*100:.1f}%  ({total_new}/{n*5})")
print(f"\nOutput: {OUTPUT_TXT}")
