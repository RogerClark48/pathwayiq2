"""
explore_career_network.py
Diagnostic: sends one cluster's job data to Haiku, gets back a node/edge
career network JSON. Console output only — no DB writes.

Run from project root with venv active:
  venv/Scripts/python.exe scripts/explore_career_network.py
"""

import json
import sqlite3
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(__file__).resolve().parent.parent / "job_roles_asset.db"

# ---------------------------------------------------------------------------
# Step 1 — Inspect cluster structure, pick largest cluster
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1 — Cluster summary")
print("=" * 60)

db = sqlite3.connect(DB_PATH)
db.row_factory = sqlite3.Row

clusters = db.execute("""
    SELECT c.cluster_id, c.name, COUNT(j.id) as job_count
    FROM clusters c
    JOIN jobs j ON j.cluster_id = c.cluster_id
    GROUP BY c.cluster_id
    ORDER BY job_count DESC
""").fetchall()

for r in clusters:
    print(f"  {r['cluster_id']:>4}  {r['name']:<45}  {r['job_count']:>3} jobs")

chosen = dict(clusters[0])
print(f"\nUsing cluster: {chosen['cluster_id']} — {chosen['name']} ({chosen['job_count']} jobs)")

# ---------------------------------------------------------------------------
# Step 2 — Pull cluster jobs
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 — Job data")
print("=" * 60)

jobs = db.execute("""
    SELECT id, title, level, salary_min, salary_max,
           entry_routes, career_prospects, progression
    FROM jobs
    WHERE cluster_id = ?
    ORDER BY level, title
""", (chosen["cluster_id"],)).fetchall()

jobs = [dict(j) for j in jobs]
db.close()

n_cp   = sum(1 for j in jobs if j["career_prospects"] and j["career_prospects"].strip())
n_prog = len(jobs) - n_cp
print(f"  {len(jobs)} jobs retrieved")
print(f"  career_prospects populated: {n_cp}")
print(f"  falling back to progression: {n_prog}")

# ---------------------------------------------------------------------------
# Step 3 — Build user message
# ---------------------------------------------------------------------------
def fmt_salary(mn, mx):
    if mn and mn > 0 and mx and mx > 0:
        return f"{mn:,}–{mx:,}"
    elif mn and mn > 0:
        return f"{mn:,}+"
    elif mx and mx > 0:
        return f"up to {mx:,}"
    return "not available"

lines = [f"CLUSTER: {chosen['cluster_id']} — {chosen['name']}", ""]

for j in jobs:
    level_str  = f"Level {j['level']}" if j["level"] else "Level unknown"
    salary_str = fmt_salary(j["salary_min"], j["salary_max"])
    lines.append(f"JOB {j['id']} | {j['title']} | {level_str} | Salary: {salary_str}")

    entry = (j["entry_routes"] or "").strip()
    lines.append(f"entry_routes: {entry or 'not available'}")

    cp   = (j["career_prospects"] or "").strip()
    prog = (j["progression"] or "").strip()

    if cp:
        lines.append(f"career_prospects: {cp}")
    elif prog:
        lines.append(f"career_prospects: not available — using progression")
        lines.append(f"progression: {prog}")
    else:
        lines.append("career_prospects: not available")

    lines.append("")

user_message = "\n".join(lines)

SYSTEM_PROMPT = """\
You are a career pathway analyst. You will be given a list of job roles within a single occupational cluster. Your task is to produce a directed network of career relationships between these roles.

Output ONLY valid JSON in this exact structure:
{
  "cluster_id": "<id>",
  "nodes": [
    {"job_id": <int>, "title": "<string>", "seniority_tier": <1|2|3|4>}
  ],
  "edges": [
    {"from_job_id": <int>, "to_job_id": <int>, "edge_type": "<upward|lateral>"}
  ]
}

Seniority tiers:
  1 = entry level (no prior experience required)
  2 = mid-level (a few years experience, technician/practitioner)
  3 = senior (specialist, experienced professional, chartered in some fields)
  4 = chartered, principal, or lead (highest tier)

Edge rules:
- Only create edges between jobs in the provided list. Do not invent roles.
- Use entry_routes and career_prospects/progression prose as authoritative signals. If a role explicitly names another role in the list, create an edge.
- Where prose is sparse, use seniority_tier and domain proximity to infer plausible edges.
- upward = the destination role typically requires more experience or qualification than the source.
- lateral = roles at similar seniority that practitioners commonly move between.
- No downward edges. The network shows where a role leads, not where it came from.
- Prefer fewer strong connections over many weak ones."""

# ---------------------------------------------------------------------------
# Step 3 — API call
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 — Haiku API call")
print("=" * 60)
print(f"  Sending {len(jobs)} jobs...")

client = Anthropic()
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=4000,
    temperature=0,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_message}],
)

raw_text = response.content[0].text.strip()
print(f"  Response: {response.usage.output_tokens} tokens")

# ---------------------------------------------------------------------------
# Step 4 — Parse and review
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4 — Results")
print("=" * 60)

try:
    # Strip markdown code fences if present
    text = raw_text
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    network = json.loads(text)
except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}")
    print("\nRaw response:")
    print(raw_text)
    raise SystemExit(1)

nodes = network.get("nodes", [])
edges = network.get("edges", [])

n_upward  = sum(1 for e in edges if e.get("edge_type") == "upward")
n_lateral = sum(1 for e in edges if e.get("edge_type") == "lateral")

print(f"\nNodes : {len(nodes)}")
print(f"Edges : {len(edges)}  (upward: {n_upward}, lateral: {n_lateral})")

# Seniority tier breakdown
from collections import Counter
tier_counts = Counter(n["seniority_tier"] for n in nodes)
print("\nSeniority tiers:")
tier_labels = {1: "entry", 2: "mid", 3: "senior", 4: "principal"}
for tier in sorted(tier_counts):
    label = tier_labels.get(tier, "unknown")
    print(f"  Tier {tier} ({label}): {tier_counts[tier]} jobs")

# Isolated nodes (zero outbound edges)
outbound = Counter(e["from_job_id"] for e in edges)
node_ids  = {n["job_id"] for n in nodes}
isolated  = [n for n in nodes if outbound[n["job_id"]] == 0]
if isolated:
    print(f"\nIsolated nodes (0 outbound edges): {len(isolated)}")
    for n in isolated:
        print(f"  [{n['job_id']}] {n['title']} (tier {n['seniority_tier']})")
else:
    print("\nNo isolated nodes.")

# Full pretty-printed JSON
print("\n" + "=" * 60)
print("FULL JSON RESPONSE")
print("=" * 60)
print(json.dumps(network, indent=2))

# Save to file
out_path = Path(__file__).resolve().parent.parent / f"cluster_{network.get('cluster_id', 'D2')}_network.json"
out_path.write_text(json.dumps(network, indent=2), encoding="utf-8")
print(f"\nSaved: {out_path}")
