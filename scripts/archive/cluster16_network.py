"""
cluster16_network.py — Generate a career pathway network for Ward medium
Cluster 16 (Electrical, electronics & telecoms) using a single Sonnet call.

Outputs to project root:
  cluster_16_network.json
  cluster_16_narrative.txt
  cluster_16_network.html

Run:
    C:\Dev\pathwayiq2\venv\Scripts\python.exe scripts/cluster16_network.py
"""

import json
import re
import sqlite3
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "job_roles_asset.db"
load_dotenv(PROJECT_ROOT / ".env")

JSON_OUT      = PROJECT_ROOT / "cluster_16_network.json"
NARRATIVE_OUT = PROJECT_ROOT / "cluster_16_narrative.txt"
HTML_OUT      = PROJECT_ROOT / "cluster_16_network.html"

# ---------------------------------------------------------------------------
# Target normalized titles
# ---------------------------------------------------------------------------
NORMALIZED_TITLES = [
    "auto electrician",
    "broadcast engineer",
    "communications engineer",
    "electrical engineer",
    "electrical engineering technician",
    "electrician",
    "electronics engineer",
    "electronics engineering technician",
    "technical sales engineer",
    "telecoms engineer",
]

# ---------------------------------------------------------------------------
# Step 1 — Pull job records, prefer NCS where both sources exist
# ---------------------------------------------------------------------------
print("Loading job records from DB...")

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

placeholders = ",".join("?" * len(NORMALIZED_TITLES))
cur.execute(f"""
    SELECT id, title, normalized_title, salary_min, salary_max,
           entry_routes, progression, overview, source
    FROM jobs
    WHERE normalized_title IN ({placeholders})
      AND overview IS NOT NULL
    ORDER BY normalized_title,
             CASE source WHEN 'ncs' THEN 0 ELSE 1 END
""", NORMALIZED_TITLES)

all_rows = cur.fetchall()
conn.close()

# Deduplicate: keep first row per normalized_title (NCS preferred by ORDER BY above)
seen = set()
jobs = []
for row in all_rows:
    nt = row["normalized_title"]
    if nt not in seen:
        seen.add(nt)
        jobs.append(dict(row))

print(f"  {len(jobs)} job records loaded ({len(all_rows)} total rows before dedup)")
for j in jobs:
    print(f"  [{j['source']:8s}] {j['title']}")

# ---------------------------------------------------------------------------
# Step 2 — Build Sonnet prompt
# ---------------------------------------------------------------------------

def format_salary(salary_min, salary_max) -> str:
    """Format salary for the user message. 0 is null sentinel."""
    has_min = salary_min is not None and salary_min > 0
    has_max = salary_max is not None and salary_max > 0
    if has_min and has_max:
        return f"GBP{salary_min:,}--GBP{salary_max:,}"
    if has_min:
        return f"GBP{salary_min:,} entry"
    return ""


SYSTEM_PROMPT = """You are a careers advisor with deep knowledge of technical and engineering career pathways.
You are analysing a cluster of related job roles to map career progression routes within that cluster.

Your task is to produce two outputs:

1. A JSON network definition describing career pathways within the cluster
2. A plain English career narrative describing the cluster as a career territory

Rules for the network definition:
- Every job in the input must appear as a node
- Edges represent natural career progressions - where a person might move from one role to another
- Edge direction: from earlier-career role to later-career role
- Use salary data as the primary indicator of seniority where available (salary_min for entry point, salary_max for ceiling)
- Use the progression and entry_routes fields as authoritative source material for edges -- if a role's progression field names another role in the cluster, that edge should exist
- Use your own career knowledge to add edges where the prose is silent but the connection is well-established
- A role may have multiple outgoing edges (career branches) and multiple incoming edges (multiple routes in)
- Keep the network honest -- do not force connections that don't exist

JSON format:
{
  "cluster_name": "Electrical, electronics & telecoms",
  "nodes": [
    {
      "id": "electrician",
      "title": "Electrician",
      "salary_min": 24000,
      "salary_max": 45000,
      "entry_level": true,
      "summary": "One sentence describing the role"
    }
  ],
  "edges": [
    {
      "from": "electrician",
      "to": "electrical_engineer",
      "label": "with degree or HNC"
    }
  ]
}

Use normalized_title (lowercase, spaces replaced with underscores) as node id.
Set entry_level: true for roles that are typical starting points (lower salary_min, entry_routes suggests apprenticeship or college entry).
The label on each edge should be a short phrase describing what enables the transition -- a qualification, experience, or specialisation.
Salary values: use integers. If salary data is missing for a role, omit the salary fields rather than guessing.

Rules for the career narrative:
- 2-3 paragraphs, plain English
- Describe the cluster as a career territory: how someone enters it, how they progress, where the branches are
- Mention salary range across the cluster (entry to ceiling)
- Understated British tone -- informative, not promotional
- No bullet points, no headers -- flowing prose only

Respond with valid JSON first, then a separator line (---), then the narrative. Nothing else."""


def build_user_message(jobs: list[dict]) -> str:
    parts = ["Here are the job records for the cluster. Analyse them and produce the network definition and narrative.\n"]
    for job in jobs:
        salary_str = format_salary(job["salary_min"], job["salary_max"])
        parts.append(f"JOB: {job['title']}")
        if salary_str:
            parts.append(f"Salary: {salary_str}")
        if job["entry_routes"]:
            parts.append(f"Entry routes: {job['entry_routes']}")
        if job["progression"]:
            parts.append(f"Progression: {job['progression']}")
        if job["overview"]:
            parts.append(f"Overview: {job['overview']}")
        parts.append("")
    return "\n".join(parts)


user_message = build_user_message(jobs)

# ---------------------------------------------------------------------------
# Step 3 — Call Sonnet
# ---------------------------------------------------------------------------
print("\nCalling Sonnet (this takes ~10 seconds)...")

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_message}],
)
raw = response.content[0].text.strip()
print("  Response received.")

# ---------------------------------------------------------------------------
# Step 4 — Parse response
# ---------------------------------------------------------------------------
# Split on separator line --- between JSON and narrative
parts = re.split(r"\n---\n", raw, maxsplit=1)
if len(parts) != 2:
    print("\nERROR: could not find --- separator in Sonnet response.")
    print("Raw response:\n")
    print(raw)
    raise SystemExit(1)

json_block, narrative = parts[0].strip(), parts[1].strip()

# Strip markdown fences if present
json_block = re.sub(r"^```(?:json)?\s*", "", json_block)
json_block = re.sub(r"\s*```$", "", json_block).strip()

try:
    network = json.loads(json_block)
except json.JSONDecodeError as e:
    print(f"\nERROR: JSON parse failed — {e}")
    print("Raw JSON block:\n")
    print(json_block)
    raise SystemExit(1)

print(f"  Parsed: {len(network.get('nodes', []))} nodes, {len(network.get('edges', []))} edges")

# Write JSON and narrative
JSON_OUT.write_text(json.dumps(network, indent=2, ensure_ascii=False), encoding="utf-8")
NARRATIVE_OUT.write_text(narrative, encoding="utf-8")
print(f"  Saved: {JSON_OUT.name}")
print(f"  Saved: {NARRATIVE_OUT.name}")

# ---------------------------------------------------------------------------
# Step 5 — Generate HTML visualisation
# ---------------------------------------------------------------------------
print("\nGenerating HTML visualisation...")

nodes = network.get("nodes", [])
edges = network.get("edges", [])

# Normalise salary_max values for node sizing (range: 20-50 px)
salary_maxes = [n["salary_max"] for n in nodes if "salary_max" in n]
sal_min_val  = min(salary_maxes) if salary_maxes else 40000
sal_max_val  = max(salary_maxes) if salary_maxes else 80000
sal_range    = max(sal_max_val - sal_min_val, 1)


def node_size(salary_max) -> int:
    if salary_max is None:
        return 28  # default mid-size when no salary data
    normalised = (salary_max - sal_min_val) / sal_range
    return int(20 + normalised * 30)


# Build vis-network node/edge arrays as JS literals
vis_nodes = []
for n in nodes:
    color  = "#4caf50" if n.get("entry_level") else "#2196f3"
    size   = node_size(n.get("salary_max"))
    title_lines = [f"<b>{n['title']}</b>"]
    if "salary_min" in n and "salary_max" in n:
        title_lines.append(f"GBP{n['salary_min']:,} - GBP{n['salary_max']:,}")
    elif "salary_min" in n:
        title_lines.append(f"from GBP{n['salary_min']:,}")
    if "summary" in n:
        title_lines.append(n["summary"])
    tooltip = "<br>".join(title_lines)

    vis_nodes.append(
        f'  {{id: {json.dumps(n["id"])}, label: {json.dumps(n["title"])}, '
        f'color: {json.dumps(color)}, size: {size}, '
        f'title: {json.dumps(tooltip)}}}'
    )

vis_edges = []
for i, e in enumerate(edges):
    vis_edges.append(
        f'  {{id: {i}, from: {json.dumps(e["from"])}, to: {json.dumps(e["to"])}, '
        f'label: {json.dumps(e.get("label", ""))}, arrows: "to"}}'
    )

cluster_name = network.get("cluster_name", "Cluster 16")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{cluster_name} — Career Pathway Network</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css">
  <style>
    body {{
      font-family: sans-serif;
      margin: 0;
      background: #f5f5f5;
    }}
    h1 {{
      font-size: 1.1rem;
      padding: 12px 20px 4px;
      margin: 0;
      color: #333;
    }}
    #legend {{
      padding: 4px 20px 8px;
      font-size: 0.8rem;
      color: #555;
    }}
    #legend span {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      margin-right: 4px;
      vertical-align: middle;
    }}
    #network {{
      width: 100%;
      height: calc(100vh - 80px);
      border-top: 1px solid #ddd;
      background: #fff;
    }}
  </style>
</head>
<body>
  <h1>{cluster_name} — Career Pathway Network</h1>
  <div id="legend">
    <span style="background:#4caf50"></span>Entry-level role &nbsp;&nbsp;
    <span style="background:#2196f3"></span>Senior / specialist role &nbsp;&nbsp;
    Node size reflects salary ceiling where known. Hover for details.
  </div>
  <div id="network"></div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
  <script>
    const nodes = new vis.DataSet([
{(",\n").join(vis_nodes)}
    ]);

    const edges = new vis.DataSet([
{(",\n").join(vis_edges)}
    ]);

    const container = document.getElementById("network");
    const data = {{ nodes, edges }};
    const options = {{
      layout: {{
        hierarchical: {{
          enabled: true,
          direction: "UD",
          sortMethod: "directed",
          levelSeparation: 120,
          nodeSpacing: 160
        }}
      }},
      nodes: {{
        shape: "dot",
        font: {{ size: 13, face: "sans-serif" }},
        borderWidth: 2,
        borderWidthSelected: 3
      }},
      edges: {{
        font: {{ size: 11, align: "middle" }},
        smooth: {{ type: "cubicBezier", forceDirection: "vertical" }},
        color: {{ color: "#888", highlight: "#333" }}
      }},
      physics: {{ enabled: false }},
      interaction: {{ hover: true, tooltipDelay: 100 }}
    }};

    new vis.Network(container, data, options);
  </script>
</body>
</html>
"""

HTML_OUT.write_text(html, encoding="utf-8")
print(f"  Saved: {HTML_OUT.name}")
print(f"\nDone.")
print(f"  {JSON_OUT}")
print(f"  {NARRATIVE_OUT}")
print(f"  {HTML_OUT}")
