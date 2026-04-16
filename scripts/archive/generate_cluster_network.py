"""
generate_cluster_network.py — Generate career pathway networks for one or more
Ward medium clusters using a single Sonnet call per cluster.

For each cluster, outputs to the project root:
  cluster_N_network.json
  cluster_N_narrative.txt
  cluster_N_network.html

Run:
    C:\Dev\pathwayiq2\venv\Scripts\python.exe scripts/generate_cluster_network.py
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

# ---------------------------------------------------------------------------
# Cluster definitions
# Each entry: (cluster_id, display_name, [normalized_titles])
# ---------------------------------------------------------------------------
CLUSTERS = [
    (
        16,
        "Electrical, electronics & telecoms",
        [
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
        ],
    ),
    (
        13,
        "Life sciences, chemistry & materials",
        [
            "analytical chemist",
            "biochemist",
            "biotechnologist",
            "chemist",
            "clinical scientist",
            "clinical scientist genomics",       # normalized: comma stripped
            "field trials officer",
            "geneticist",
            "laboratory technician",
            "materials engineer",
            "materials technician",
            "medicinal chemist",
            "microbiologist",
            "nanotechnologist",
            "pharmacologist",
            "physicist",
            "plant breedergeneticist",           # normalized: slash stripped
            "research scientist",
            "scientific laboratory technician",
            "teaching laboratory technician",
        ],
    ),
    (
        1,
        "Software development",
        [
            "app developer",
            "applications developer",
            "computer games tester",
            "software developer",
            "software engineer",
            "software tester",
            "test lead",
            "web developer",
        ],
    ),
    (
        2,
        "Games & creative digital",
        [
            "computer games developer",
            "e-learning developer",
            "game artist",
            "game designer",
            "game developer",
            "medical illustrator",
            "multimedia programmer",
            "multimedia specialist",
            "technical author",
            "ux designer",
        ],
    ),
    (
        3,
        "AI & data science",
        [
            "artificial intelligence (ai) engineer",
            "data analyst",
            "data analyst-statistician",
            "data scientist",
            "machine learning engineer",
            "operational researcher",
            "statistician",
        ],
    ),
    (
        4,
        "Cyber security",
        [
            "cyber intelligence officer",
            "cyber security analyst",
            "forensic computer analyst",
            "it security co-ordinator",
            "penetration tester",
        ],
    ),
    (
        5,
        "IT systems & infrastructure",
        [
            "application analyst",
            "database administrator",
            "it consultant",
            "it support technician",
            "information systems manager",
            "network engineer",
            "network manager",
            "solutions architect",
            "systems analyst",
            "technical architect",
        ],
    ),
    (
        6,
        "Building surveying & compliance",
        [
            "acoustic consultant",
            "acoustics consultant",
            "building control officer",
            "building control surveyor",
            "building surveyor",
            "fire safety engineer",
        ],
    ),
    (
        7,
        "Construction & civil engineering",
        [
            "architectural technician",
            "architectural technologist",
            "building site inspector",
            "building technician",
            "cad technician",
            "civil engineer",
            "civil engineering technician",
            "construction contracts manager",
            "construction manager",
            "construction site supervisor",
            "consulting civil engineer",
            "contracting civil engineer",
            "estimator",
            "quantity surveyor",
            "site engineer",
            "surveying technician",
        ],
    ),
    (
        8,
        "GIS & land surveying",
        [
            "geographical information systems officer",
            "geospatial  technician",    # double space in normalized_title
            "hydrographic surveyor",
            "land surveyor",
            "landgeomatics surveyor",    # normalized: slash stripped
        ],
    ),
    (
        9,
        "Earth & geo sciences",
        [
            "drilling engineer",
            "engineering geologist",
            "environmental engineer",
            "geochemist",
            "geophysicist",
            "geoscientist",
            "geotechnical engineer",
            "geotechnician",
            "hydrogeologist",
            "hydrologist",
            "minerals surveyor",
            "mining engineer",
            "petroleum engineer",
            "quarry engineer",
            "seismologist",
            "water engineer",
            "water quality scientist",
        ],
    ),
    (
        10,
        "Maritime & naval",
        [
            "armed forces technical officer",
            "marine engineer",
            "marine engineering technician",
            "merchant navy deck officer",
            "merchant navy engineering officer",
            "merchant navy officer",
            "naval architect",
        ],
    ),
    (
        11,
        "Aviation, space & weather",
        [
            "air accident investigator",
            "air traffic controller",
            "airline pilot",
            "astronaut",
            "astronomer",
            "climate scientist",
            "drone pilot",
            "meteorologist",
            "raf aviator",
        ],
    ),
    (
        12,
        "Medical physics & nuclear",
        [
            "biomedical engineer",
            "clinical engineer",
            "clinical scientist medical physics",  # normalized: comma stripped
            "clinical technologist",
            "medical physicist",
            "nuclear engineer",
            "nuclear technician",
            "prosthetist and orthotist",
            "radiation protection practitioner",
        ],
    ),
    (
        14,
        "Energy & renewables",
        [
            "building services engineer",
            "commercial energy assessor",
            "energy engineer",
            "energy manager",
            "heat pump engineer",
            "heating and ventilation engineer",
            "refrigeration and air-conditioning installer",
            "renewable energy engineer",
            "thermal insulation engineer",
        ],
    ),
    (
        "15a",
        "Agricultural & land-based engineering",
        [
            "agricultural engineer",
            "agricultural engineering technician",
            "landbased engineer",                  # normalized: hyphen stripped
        ],
    ),
    (
        "15b",
        "Plant & maintenance engineering",
        [
            "construction plant mechanic",
            "construction plant operator",
            "electricity distribution worker",
            "electricity generation worker",
            "engineering maintenance technician",
            "helicopter engineer",
            "lift engineer",
            "maintenance engineer",
            "maintenance fitter",
            "pipe fitter",
            "rolling stock engineering technician",
            "signalling technician",
            "wind turbine technician",
        ],
    ),
    (
        17,
        "Precision & metalwork manufacturing",
        [
            "3d printing technician",
            "cnc machinist",
            "engineering operative",
            "foundry moulder",
            "metallurgist",
            "metrologist",
            "non-destructive testing technician",
            "toolmaker",
            "welder",
        ],
    ),
    (
        18,
        "Mechanical, aerospace & automotive",
        [
            "aerospace engineer",
            "aerospace engineering technician",
            "automotive engineer",
            "car manufacturing worker",
            "design and development engineer",
            "design engineer",
            "engineering construction technician",
            "mechanical engineer",
            "mechanical engineering technician",
            "motorsport engineer",
            "product designer",
            "robotics engineer",
            "steel erector",
            "structural engineer",
        ],
    ),
    (
        19,
        "IT project & production management",
        [
            "digital delivery manager",
            "it project manager",
            "production manager",
            "production manager (manufacturing)",
        ],
    ),
    (
        20,
        "Chemical & process engineering",
        [
            "chemical engineer",
            "chemical engineering technician",
            "chemical plant process operator",
            "control and instrumentation engineer",
            "food technologist",
            "manufacturing engineer",
            "manufacturing systems engineer",
            "packaging technologist",
            "productprocess development scientist",  # normalized: slash stripped
        ],
    ),
]

# Clusters already generated — skip these
SKIP_CLUSTERS = {13, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18, 19, 20}

# ---------------------------------------------------------------------------
# Sonnet system prompt (updated: no bidirectional edges, no "cluster" word)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a careers advisor with deep knowledge of technical and engineering career pathways.
You are analysing a set of related job roles to map career progression routes within that career area.

Your task is to produce two outputs:

1. A JSON network definition describing career pathways
2. A plain English career narrative describing the career area

Rules for the network definition:
- Every job in the input must appear as a node
- Edges represent natural career progressions -- where a person might realistically move from one role to another over time
- Edges are strictly directional: from earlier-career role to later-career role only. If two roles are lateral peers with no clear seniority difference, do NOT create edges between them -- describe the lateral relationship in the narrative instead
- Use salary data as the primary indicator of seniority where available
- Use the progression and entry_routes fields as authoritative source material for edges -- if a role's progression field names another role in the input set, that edge must exist
- Use your own career knowledge to add edges where the prose is silent but the progression is well-established in practice
- A role may have multiple outgoing edges (career branches) and multiple incoming edges (multiple routes in)
- Keep the network honest -- do not force connections that do not exist

JSON format:
{
  "career_area": "<name of the career area>",
  "nodes": [
    {
      "id": "<normalized_title with underscores>",
      "title": "<display title>",
      "salary_min": <integer or omit if unknown>,
      "salary_max": <integer or omit if unknown>,
      "entry_level": <true if typical starting point, false otherwise>,
      "summary": "<one sentence describing the role>"
    }
  ],
  "edges": [
    {
      "from": "<node id>",
      "to": "<node id>",
      "label": "<short phrase describing what enables this transition>"
    }
  ]
}

Use normalized_title (lowercase, spaces replaced with underscores) as node id.
Set entry_level: true for roles that are typical starting points (lower salary, entry_routes suggests apprenticeship or college entry).
Edge labels should be short phrases describing what enables the transition -- a qualification, experience, or specialisation.
Salary values: integers only. Omit salary fields entirely if unknown -- do not guess.

Rules for the career narrative:
- 2-3 paragraphs, plain English, flowing prose
- Describe the career area: how someone enters it, how they progress, where the branches are
- Mention salary range across the area (entry to ceiling) where data is available
- Where roles are lateral peers rather than sequential, describe those relationships here rather than as edges in the network
- Understated British tone -- informative, not promotional
- No bullet points, no headers
- Never use the word "cluster" -- refer to "this field", "this area", "these careers", or the career area name

Respond with valid JSON first, then a separator line containing only ---, then the narrative.
Nothing else before, between or after."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jobs(normalized_titles: list[str]) -> list[dict]:
    """Pull job records from DB, NCS preferred over Prospects per normalized_title."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    placeholders = ",".join("?" * len(normalized_titles))
    cur.execute(f"""
        SELECT id, title, normalized_title, salary_min, salary_max,
               entry_routes, progression, overview, source
        FROM jobs
        WHERE normalized_title IN ({placeholders})
          AND overview IS NOT NULL
        ORDER BY normalized_title,
                 CASE source WHEN 'ncs' THEN 0 ELSE 1 END
    """, normalized_titles)

    all_rows = cur.fetchall()
    conn.close()

    seen, jobs = set(), []
    for row in all_rows:
        nt = row["normalized_title"]
        if nt not in seen:
            seen.add(nt)
            jobs.append(dict(row))

    return jobs, len(all_rows)


def format_salary(salary_min, salary_max) -> str:
    has_min = salary_min is not None and salary_min > 0
    has_max = salary_max is not None and salary_max > 0
    if has_min and has_max:
        return f"GBP{salary_min:,}--GBP{salary_max:,}"
    if has_min:
        return f"GBP{salary_min:,} entry"
    return ""


def build_user_message(jobs: list[dict]) -> str:
    parts = [
        "Here are the job records for this career area. Analyse them and produce "
        "the network definition and narrative.\n"
    ]
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


def parse_sonnet_response(raw: str) -> tuple[dict, str]:
    """
    Split on --- separator. Returns (network_dict, narrative_str).
    Raises ValueError with raw response on parse failure.
    """
    parts = re.split(r"\n---\n", raw, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"No --- separator found in response.\n\nRaw:\n{raw}")

    json_block = parts[0].strip()
    narrative  = parts[1].strip()

    # Strip markdown fences if present
    json_block = re.sub(r"^```(?:json)?\s*", "", json_block)
    json_block = re.sub(r"\s*```$", "", json_block).strip()

    try:
        network = json.loads(json_block)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}\n\nJSON block:\n{json_block}")

    return network, narrative


def generate_html(network: dict) -> str:
    nodes = network.get("nodes", [])
    edges = network.get("edges", [])
    career_area = network.get("career_area", "Career Pathway Network")

    salary_maxes = [n["salary_max"] for n in nodes if "salary_max" in n]
    sal_min_val  = min(salary_maxes) if salary_maxes else 40000
    sal_max_val  = max(salary_maxes) if salary_maxes else 80000
    sal_range    = max(sal_max_val - sal_min_val, 1)

    def node_size(salary_max) -> int:
        if salary_max is None:
            return 28
        return int(20 + ((salary_max - sal_min_val) / sal_range) * 30)

    vis_nodes = []
    for n in nodes:
        color = "#4caf50" if n.get("entry_level") else "#2196f3"
        size  = node_size(n.get("salary_max"))
        tip_lines = [f"<b>{n['title']}</b>"]
        if "salary_min" in n and "salary_max" in n:
            tip_lines.append(f"GBP{n['salary_min']:,} - GBP{n['salary_max']:,}")
        elif "salary_min" in n:
            tip_lines.append(f"from GBP{n['salary_min']:,}")
        if "summary" in n:
            tip_lines.append(n["summary"])
        tooltip = "<br>".join(tip_lines)

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

    nodes_js = ",\n".join(vis_nodes)
    edges_js = ",\n".join(vis_edges)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{career_area} - Career Pathway Network</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css">
  <style>
    body {{ font-family: sans-serif; margin: 0; background: #f5f5f5; }}
    h1 {{ font-size: 1.1rem; padding: 12px 20px 4px; margin: 0; color: #333; }}
    #legend {{ padding: 4px 20px 8px; font-size: 0.8rem; color: #555; }}
    #legend span {{
      display: inline-block; width: 12px; height: 12px;
      border-radius: 50%; margin-right: 4px; vertical-align: middle;
    }}
    #network {{
      width: 100%; height: calc(100vh - 80px);
      border-top: 1px solid #ddd; background: #fff;
    }}
  </style>
</head>
<body>
  <h1>{career_area} - Career Pathway Network</h1>
  <div id="legend">
    <span style="background:#4caf50"></span>Entry-level role &nbsp;&nbsp;
    <span style="background:#2196f3"></span>Senior / specialist role &nbsp;&nbsp;
    Node size reflects salary ceiling where known. Hover for details.
  </div>
  <div id="network"></div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
  <script>
    const nodes = new vis.DataSet([
{nodes_js}
    ]);
    const edges = new vis.DataSet([
{edges_js}
    ]);
    const container = document.getElementById("network");
    const options = {{
      layout: {{
        hierarchical: {{
          enabled: true, direction: "UD", sortMethod: "directed",
          levelSeparation: 130, nodeSpacing: 170
        }}
      }},
      nodes: {{
        shape: "dot",
        font: {{ size: 13, face: "sans-serif" }},
        borderWidth: 2, borderWidthSelected: 3
      }},
      edges: {{
        font: {{ size: 11, align: "middle" }},
        smooth: {{ type: "cubicBezier", forceDirection: "vertical" }},
        color: {{ color: "#888", highlight: "#333" }}
      }},
      physics: {{ enabled: false }},
      interaction: {{ hover: true, tooltipDelay: 100 }}
    }};
    new vis.Network(container, {{ nodes, edges }}, options);
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
client = anthropic.Anthropic()

for cluster_id, display_name, normalized_titles in CLUSTERS:
    if cluster_id in SKIP_CLUSTERS:
        print(f"\n[Cluster {cluster_id} — {display_name}] already complete, skipping.")
        continue
    print(f"\n[Cluster {cluster_id} -- {display_name}] generating...")

    # Load jobs
    jobs, total_rows = load_jobs(normalized_titles)
    print(f"  {len(jobs)} records loaded ({total_rows} rows before dedup)")
    for j in jobs:
        print(f"  [{j['source']:8s}] {j['title']}")

    missing = set(normalized_titles) - {j["normalized_title"] for j in jobs}
    if missing:
        print(f"  WARNING: no DB record found for: {', '.join(sorted(missing))}")

    # Call Sonnet
    print(f"\n  Calling Sonnet...")
    user_message = build_user_message(jobs)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text.strip()

    # Parse
    try:
        network, narrative = parse_sonnet_response(raw)
    except ValueError as e:
        print(f"\n  ERROR parsing Sonnet response:\n{e}")
        print("  Skipping this cluster.")
        continue

    n_nodes = len(network.get("nodes", []))
    n_edges = len(network.get("edges", []))
    print(f"  Parsed: {n_nodes} nodes, {n_edges} edges")

    # Write outputs
    json_path      = PROJECT_ROOT / f"cluster_{cluster_id}_network.json"
    narrative_path = PROJECT_ROOT / f"cluster_{cluster_id}_narrative.txt"
    html_path      = PROJECT_ROOT / f"cluster_{cluster_id}_network.html"

    json_path.write_text(json.dumps(network, indent=2, ensure_ascii=False), encoding="utf-8")
    narrative_path.write_text(narrative, encoding="utf-8")
    html_path.write_text(generate_html(network), encoding="utf-8")

    print(f"[Cluster {cluster_id}] done -- {n_nodes} nodes, {n_edges} edges")

print(f"\n{'=' * 60}")
print("Done.")
