"""
explore_network_snippet.py
Generate a small standalone HTML showing a single job's immediate
upward and lateral paths from the career network tables.

Usage:
  venv/Scripts/python.exe scripts/explore_network_snippet.py <job_id>
"""

import json
import sqlite3
import sys
from pathlib import Path

DB_PATH     = Path(__file__).resolve().parent.parent / "job_roles_asset.db"
SCRIPTS_DIR = Path(__file__).resolve().parent

TIER_LABELS  = {1:"Entry", 2:"Mid-entry", 3:"Mid", 4:"Senior", 5:"Principal", 6:"Consulting/contracting"}
TIER_COLOURS = {1:"#4caf50", 2:"#8bc34a", 3:"#2196f3", 4:"#1565c0", 5:"#9c27b0", 6:"#e65100"}


def main():
    if len(sys.argv) < 2:
        print("Usage: explore_network_snippet.py <job_id>")
        sys.exit(1)

    job_id = int(sys.argv[1])

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    # Current job
    node = db.execute(
        "SELECT cluster_id, seniority_tier FROM career_network_nodes WHERE job_id=? LIMIT 1",
        (job_id,)
    ).fetchone()
    if not node:
        print(f"job_id {job_id} not found in career_network_nodes")
        sys.exit(1)

    cluster_id = node["cluster_id"]
    self_tier  = node["seniority_tier"]

    self_title = db.execute(
        "SELECT title FROM jobs WHERE id=? LIMIT 1", (job_id,)
    ).fetchone()["title"]

    # Load full edge table for this cluster into memory, then BFS from job_id
    all_edges = db.execute(
        "SELECT from_job_id, to_job_id, edge_type FROM career_network_edges WHERE cluster_id=?",
        (cluster_id,)
    ).fetchall()

    # Build adjacency: from_job_id -> list of (to_job_id, edge_type)
    adj = {}
    for e in all_edges:
        adj.setdefault(e["from_job_id"], []).append((e["to_job_id"], e["edge_type"]))

    # BFS — collect all reachable nodes and the edges that connect them
    visited_nodes = {job_id}
    reachable_edges = []   # (from, to, edge_type)
    queue = [job_id]
    while queue:
        current = queue.pop(0)
        for dest, etype in adj.get(current, []):
            reachable_edges.append((current, dest, etype))
            if dest not in visited_nodes:
                visited_nodes.add(dest)
                queue.append(dest)

    # Fetch node info for all visited nodes
    node_info = {}
    for nid in visited_nodes:
        tier_row  = db.execute(
            "SELECT seniority_tier FROM career_network_nodes WHERE job_id=? AND cluster_id=?",
            (nid, cluster_id)
        ).fetchone()
        title_row = db.execute("SELECT title FROM jobs WHERE id=? LIMIT 1", (nid,)).fetchone()
        node_info[nid] = {
            "title": title_row["title"] if title_row else f"Job {nid}",
            "tier":  tier_row["seniority_tier"] if tier_row else self_tier,
        }

    # Direct edges (one hop) for console summary
    edges = [(dest, etype) for _, dest, etype in reachable_edges if _ == job_id]
    dest_info = {dest: {**node_info[dest], "edge_type": etype} for dest, etype in edges}

    db.close()

    # ── Console summary ──
    print(f"\nJob {job_id}: {self_title} (Tier {self_tier}: {TIER_LABELS.get(self_tier,'')}, cluster {cluster_id})")
    if not edges:
        print("  (no outbound edges — terminal node)")
    for tid, etype in edges:
        d     = dest_info[tid]
        arrow = "->" if etype == "upward" else "<->"
        print(f"  {arrow} {etype}: {d['title']} ({tid}, Tier {d['tier']})")

    # ── Build vis-network HTML ──
    vis_nodes = []
    for nid, info in node_info.items():
        tier   = info["tier"]
        is_self = nid == job_id
        colour = "white" if is_self else TIER_COLOURS.get(tier, "#888")
        border = "#333"  if is_self else TIER_COLOURS.get(tier, "#888")
        size   = 22      if is_self else 18
        tip    = f"<b>{info['title']}</b><br>Tier {tier}: {TIER_LABELS.get(tier, '')}<br>ID: {nid}"
        vis_nodes.append(
            f'  {{id:{nid}, label:{json.dumps(info["title"])}, '
            f'color:{{background:{json.dumps(colour)}, border:{json.dumps(border)}}}, '
            f'size:{size}, title:{json.dumps(tip)}}}'
        )

    vis_edges = []
    for i, (efrom, eto, etype) in enumerate(reachable_edges):
        dash = "false" if etype == "upward" else "true"
        vis_edges.append(
            f'  {{id:{i}, from:{efrom}, to:{eto}, '
            f'arrows:"to", dashes:{dash}, title:{json.dumps(etype)}}}'
        )

    nodes_js = ",\n".join(vis_nodes)
    edges_js = ",\n".join(vis_edges)

    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:12px">'
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{c};margin-right:4px"></span>{TIER_LABELS[t]}</span>'
        for t, c in TIER_COLOURS.items()
    )

    page_title = f"{self_title} — career paths"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{page_title}</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css">
  <style>
    body {{ font-family: sans-serif; margin:0; background:#f5f5f5; }}
    h1 {{ font-size:1.1rem; padding:12px 20px 4px; margin:0; color:#333; }}
    #meta {{ padding:2px 20px 4px; font-size:0.85rem; color:#555; }}
    #legend {{ padding:2px 20px 6px; font-size:0.8rem; color:#555; }}
    #note {{ padding:0 20px 6px; font-size:0.75rem; color:#888; }}
    #network {{ width:100%; height:calc(100vh - 110px); border-top:1px solid #ddd; background:#fff; }}
  </style>
</head>
<body>
  <h1>{page_title}</h1>
  <div id="meta">Cluster {cluster_id} &nbsp;|&nbsp; Tier {self_tier}: {TIER_LABELS.get(self_tier, '')} &nbsp;|&nbsp; Job ID: {job_id}</div>
  <div id="legend">{legend}</div>
  <div id="note">White node = current job &nbsp; Solid arrow = upward &nbsp; Dashed = lateral</div>
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
      layout: {{ hierarchical: {{ enabled:true, direction:"UD", sortMethod:"directed", levelSeparation:120, nodeSpacing:180 }} }},
      nodes: {{ shape:"dot", font:{{ size:13, face:"sans-serif" }}, borderWidth:2 }},
      edges: {{ font:{{ size:11, align:"middle" }}, smooth:{{ type:"cubicBezier", forceDirection:"vertical" }}, color:{{ color:"#888", highlight:"#333" }} }},
      physics: {{ enabled:false }},
      interaction: {{ hover:true, tooltipDelay:100 }}
    }};
    new vis.Network(container, {{ nodes, edges }}, options);
  </script>
</body>
</html>"""

    out_path = SCRIPTS_DIR / f"snippet_job_{job_id}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\nHTML written: {out_path}")


if __name__ == "__main__":
    main()
