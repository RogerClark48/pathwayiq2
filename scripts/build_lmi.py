"""
build_lmi.py — Create lmi.db, insert 8 GM cluster records, map 274 carded jobs to clusters via Haiku.

Usage:
    python scripts/build_lmi.py
"""

import json
import os
import sqlite3
import sys
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LMI_DB       = os.path.join(BASE, "lmi.db")
JOBS_DB      = os.path.join(BASE, "job_roles_asset.db")
FF_DB        = os.path.join(BASE, "futurefinder.sqlite")
HAIKU_MODEL  = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id      INTEGER PRIMARY KEY,
    region          TEXT NOT NULL,
    cluster_name    TEXT NOT NULL,
    tier            TEXT NOT NULL,
    description     TEXT NOT NULL,
    anchor_employers TEXT,
    geography       TEXT,
    cross_cutting   INTEGER NOT NULL DEFAULT 0,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS role_clusters (
    job_id      INTEGER NOT NULL,
    cluster_id  INTEGER NOT NULL,
    is_primary  INTEGER NOT NULL DEFAULT 0,
    reasoning   TEXT,
    confidence  TEXT,
    PRIMARY KEY (job_id, cluster_id)
);
"""

# ---------------------------------------------------------------------------
# Cluster records (verbatim)
# ---------------------------------------------------------------------------

CLUSTERS = [
    {
        "cluster_id": 1,
        "region": "Greater Manchester",
        "cluster_name": "Advanced Manufacturing & Materials",
        "tier": "frontier",
        "description": "GM's industrial base spanning food & drink production, special-purpose machinery, aerospace/defence and automotive supply, with growing materials/graphene strength.",
        "anchor_employers": json.dumps(["Heineken UK", "McVitie's/Pladis", "Kellogg's", "Mettler-Toledo Safeline", "BOBST", "BAE Systems"]),
        "geography": "Trafford Park heartland; manufacturing demand also in Wigan, Bolton, Chorley, Stockport",
        "cross_cutting": 0,
        "notes": "Welder/fabricator-type roles also map to Construction. Some materials/graphene work overlaps Health Innovation and Digital.",
    },
    {
        "cluster_id": 2,
        "region": "Greater Manchester",
        "cluster_name": "Digital, Cyber & AI",
        "tier": "frontier",
        "description": "One of the UK's largest tech ecosystems outside London — software, B2B SaaS, AI/data, fintech, e-commerce and a fast-growing cyber sub-cluster. More than 10,000 digital and tech companies, including 1,500 high-growth firms.",
        "anchor_employers": json.dumps(["Autotrader", "IBM", "Roku", "Booking.com", "Accenture", "Peak AI", "ANS Group", "CreateFuture", "GCHQ's Manchester cyber hub"]),
        "geography": "Circle Square and the city-centre core, but spread across neighbourhoods and out to Wigan; cyber anchored at the Digital Security Hub",
        "cross_cutting": 1,
        "notes": "Blurs heavily with Creative Industries (createch, digital media) and increasingly Health Innovation (healthtech). Hybrid-management roles often map here as primary.",
    },
    {
        "cluster_id": 3,
        "region": "Greater Manchester",
        "cluster_name": "Health Innovation & Life Sciences",
        "tier": "frontier",
        "description": "World-class precision-medicine, diagnostics, genomics and medtech cluster built around the NHS–university campus. The research/innovation end of health, distinct from care delivery.",
        "anchor_employers": json.dumps(["QIAGEN", "APIS Assay Technologies", "Takagi", "Hologic", "Yourgene Health", "Lonza", "Proteintech", "Chiesi", "Manchester University NHS Foundation Trust", "University of Manchester"]),
        "geography": "Citylabs / Oxford Road clinical-academic campus; Bruntwood SciTech campuses",
        "cross_cutting": 0,
        "notes": "Healthcare science assistant maps here, not to Health & Social Care. Diagnostics/AI overlaps Digital.",
    },
    {
        "cluster_id": 4,
        "region": "Greater Manchester",
        "cluster_name": "Creative Industries",
        "tier": "frontier",
        "description": "UK's leading creative cluster outside London — broadcast, film/TV, animation, advertising, design, music and createtech. 88,000 people in creative and digital companies, a £5bn ecosystem.",
        "anchor_employers": json.dumps(["BBC", "ITV (Coronation Street production centre)", "Dock10", "RED", "Carbon Creative"]),
        "geography": "Four micro-clusters — MediaCity (Salford/west); east (Space Studios, The Sharp Project); city-centre indies (Northern Quarter–Spinningfields); St John's/Enterprise City",
        "cross_cutting": 0,
        "notes": "Geography is itself the key user info here. Blurs with Digital.",
    },
    {
        "cluster_id": 5,
        "region": "Greater Manchester",
        "cluster_name": "Clean Growth / Net Zero",
        "tier": "frontier",
        "description": "Activity driven by GM's net-zero-by-2038 target — renewables, heat networks, hydrogen, and a very large building-retrofit programme. A greener economy is projected to create or secure around 256,000 jobs by 2038, 90,000 in retrofit alone.",
        "anchor_employers": json.dumps(["SSE", "E.ON", "Energy Systems Catapult", "University of Salford Energy House Laboratories"]),
        "geography": "Distributed across Manchester, Rochdale, Salford, Stockport, Wigan; retrofit wherever there is housing stock",
        "cross_cutting": 1,
        "notes": "Largely an overlay — much of its employment is delivered by Construction (retrofit) and Manufacturing employers. Energy assessor maps here cleanly.",
    },
    {
        "cluster_id": 6,
        "region": "Greater Manchester",
        "cluster_name": "Health & Social Care",
        "tier": "foundational",
        "description": "The care-delivery end of health — NHS acute, mental health and community trusts plus the large adult social care sector. High-volume foundational employment.",
        "anchor_employers": json.dumps(["Manchester University NHS Foundation Trust", "Greater Manchester Mental Health NHS Foundation Trust", "Northern Care Alliance"]),
        "geography": "Across all ten boroughs — employment follows population, not a single hub",
        "cross_cutting": 1,
        "notes": "Care worker, social work assistant map here. Distinct from Health Innovation. Devolved £6bn health-and-care budget is a useful framing point.",
    },
    {
        "cluster_id": 7,
        "region": "Greater Manchester",
        "cluster_name": "Construction & Built Environment",
        "tier": "supporting",
        "description": "Sustained development, regeneration and housing activity, reinforced by net-zero building targets.",
        "anchor_employers": json.dumps(["Graham Construction", "Renaker", "Russell WBHO", "Peel L&P", "Property Alliance Group"]),
        "geography": "Trafford Park and regeneration corridors (Salford Quays, Ancoats, city centre); large schemes across all boroughs",
        "cross_cutting": 1,
        "notes": "Absorbs welders/fabricators and the retrofit overlay from Clean Growth.",
    },
    {
        "cluster_id": 8,
        "region": "Greater Manchester",
        "cluster_name": "Business, Professional & Public Services",
        "tier": "supporting",
        "description": "The cross-cutting employment base — finance, professional services, administration, management and the public sector. The home for roles defined by a function performed across all sectors.",
        "anchor_employers": json.dumps(["Large GM-wide employers across finance, professional services and the public sector (local authorities, universities, GMCA)"]),
        "geography": "City-centre professional core (e.g. Spinningfields) but genuinely everywhere",
        "cross_cutting": 1,
        "notes": "Catch-all for sector-agnostic roles — facilities manager, supervisor, training officer, intelligence analyst. For these, breadth IS the message — phrase output as 'needed across many sectors' rather than naming a false home.",
    },
]

# ---------------------------------------------------------------------------
# Haiku tool definition
# ---------------------------------------------------------------------------

ASSIGN_TOOL = {
    "name": "assign_clusters",
    "description": "Assign this job role to 1-3 Greater Manchester economy clusters.",
    "input_schema": {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "cluster_id":  {"type": "integer", "minimum": 1, "maximum": 8},
                        "is_primary":  {"type": "integer", "enum": [0, 1]},
                        "reasoning":   {"type": "string"},
                        "confidence":  {"type": "string", "enum": ["high", "review"]},
                    },
                    "required": ["cluster_id", "is_primary", "reasoning", "confidence"],
                },
            }
        },
        "required": ["assignments"],
    },
}

SYSTEM = """\
You are a labour market analyst mapping job roles to Greater Manchester economic clusters.

## The 8 clusters

1 Advanced Manufacturing & Materials (frontier) — food/drink production, machinery, aerospace/defence, automotive supply, graphene/materials. Trafford Park heartland plus Wigan/Bolton/Chorley/Stockport.

2 Digital, Cyber & AI (frontier, cross-cutting) — software, SaaS, AI/data, fintech, e-commerce, cyber. City-centre core, Circle Square, Digital Security Hub. Note: blurs with Creative Industries (createch) and Health Innovation (healthtech).

3 Health Innovation & Life Sciences (frontier) — precision medicine, diagnostics, genomics, medtech. Oxford Road clinical campus, Bruntwood SciTech. Research/innovation end of health only — NOT care delivery. Healthcare science assistant maps here.

4 Creative Industries (frontier) — broadcast, film/TV, animation, advertising, design, music, createtech. MediaCity; Space Studios; Northern Quarter; St John's/Enterprise City.

5 Clean Growth / Net Zero (frontier, cross-cutting) — renewables, heat networks, hydrogen, building retrofit. Distributed across GM. Energy assessor maps here cleanly. Largely delivered by Construction and Manufacturing employers.

6 Health & Social Care (foundational, cross-cutting) — NHS acute/mental health/community trusts, adult social care. Care worker, social work assistant map here. Distinct from Health Innovation.

7 Construction & Built Environment (supporting, cross-cutting) — development, regeneration, housing, net-zero retrofit. Trafford Park, Salford Quays, Ancoats; schemes across all boroughs. Absorbs welders/fabricators.

8 Business, Professional & Public Services (supporting, cross-cutting) — finance, professional services, administration, management, public sector. Catch-all for sector-agnostic roles (facilities manager, supervisor, training officer). For these, breadth IS the message.

## Rules

- Assign 1–3 clusters. Do not force three; do not pad.
- Exactly one assignment must have is_primary = 1.
- confidence = "high" for clear calls; "review" for genuinely ambiguous roles (e.g. sector-agnostic functions, roles that sit at a cluster boundary).
- reasoning = one concise sentence explaining the primary cluster pick (or the ambiguity for review cases).
"""


def get_carded_job_ids() -> list[int]:
    conn = sqlite3.connect(FF_DB)
    rows = conn.execute(
        "SELECT curated_jobs FROM course_career_pathways WHERE curated_jobs IS NOT NULL"
    ).fetchall()
    conn.close()
    ids = set()
    for (cj,) in rows:
        for entry in json.loads(cj):
            ids.add(entry["job_id"])
    return sorted(ids)


def get_job_content(job_ids: list[int]) -> dict[int, dict]:
    conn = sqlite3.connect(JOBS_DB)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(job_ids))
    rows = conn.execute(
        f"SELECT id, title, overview, typical_duties FROM jobs WHERE id IN ({placeholders})",
        job_ids,
    ).fetchall()
    conn.close()
    return {r["id"]: dict(r) for r in rows}


def map_job(client: anthropic.Anthropic, job: dict) -> list[dict] | None:
    title    = job.get("title", "")
    overview = (job.get("overview") or "")[:400]
    duties   = (job.get("typical_duties") or "")[:300]
    content  = f"Title: {title}\nOverview: {overview}\nTypical duties: {duties}"

    try:
        resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=300,
            system=SYSTEM,
            tools=[ASSIGN_TOOL],
            tool_choice={"type": "tool", "name": "assign_clusters"},
            messages=[{"role": "user", "content": content}],
        )
        tool_use = next((b for b in resp.content if b.type == "tool_use"), None)
        if not tool_use:
            return None
        assignments = tool_use.input.get("assignments", [])
        # Validate: exactly one primary
        primaries = [a for a in assignments if a.get("is_primary") == 1]
        if len(primaries) != 1:
            # Force the first assignment to be primary
            for a in assignments:
                a["is_primary"] = 0
            assignments[0]["is_primary"] = 1
        return assignments
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        return None


def main():
    client = anthropic.Anthropic()

    # --- Step 1: Create schema ---
    print("Creating lmi.db schema...", flush=True)
    if os.path.exists(LMI_DB):
        os.remove(LMI_DB)
    conn = sqlite3.connect(LMI_DB)
    conn.executescript(SCHEMA)
    conn.commit()

    # --- Step 2: Insert clusters ---
    print("Inserting 8 cluster records...", flush=True)
    conn.executemany(
        """INSERT INTO clusters
           (cluster_id, region, cluster_name, tier, description,
            anchor_employers, geography, cross_cutting, notes)
           VALUES (:cluster_id, :region, :cluster_name, :tier, :description,
                   :anchor_employers, :geography, :cross_cutting, :notes)""",
        CLUSTERS,
    )
    conn.commit()
    print(f"  {conn.execute('SELECT COUNT(*) FROM clusters').fetchone()[0]} clusters inserted.", flush=True)

    # --- Step 3: Collect jobs ---
    print("Collecting carded job IDs...", flush=True)
    job_ids  = get_carded_job_ids()
    job_data = get_job_content(job_ids)
    print(f"  {len(job_ids)} job IDs — {len(job_data)} found in jobs DB.", flush=True)

    # --- Step 4: LLM mapping pass ---
    print(f"Running Haiku mapping pass ({len(job_ids)} jobs)...", flush=True)
    inserted = 0
    failed   = 0

    for i, jid in enumerate(job_ids, 1):
        job = job_data.get(jid)
        if not job:
            print(f"  [{i}/{len(job_ids)}] job {jid} — not in jobs DB, skipping", flush=True)
            failed += 1
            continue

        assignments = map_job(client, job)
        if not assignments:
            print(f"  [{i}/{len(job_ids)}] job {jid} ({job['title']}) — LLM failed, skipping", flush=True)
            failed += 1
            continue

        for a in assignments:
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO role_clusters (job_id, cluster_id, is_primary, reasoning, confidence) VALUES (?,?,?,?,?)",
                    (jid, a["cluster_id"], a["is_primary"], a.get("reasoning"), a.get("confidence")),
                )
            except Exception as e:
                print(f"  DB insert error job {jid}: {e}", flush=True)

        conn.commit()

        primary = next(a for a in assignments if a["is_primary"] == 1)
        clusters_str = "+".join(str(a["cluster_id"]) for a in assignments)
        print(
            f"  [{i:>3}/{len(job_ids)}] {job['title'][:40]:<40} -> {clusters_str}  [{primary['confidence']}]",
            flush=True,
        )

        # Polite rate limiting
        if i % 10 == 0:
            time.sleep(0.5)

    print(f"\nDone. {inserted + (len(job_ids) - failed - inserted)} mapped, {failed} skipped.", flush=True)

    # --- Step 5: Summary ---
    print("\n=== SUMMARY ===", flush=True)

    rows = conn.execute("""
        SELECT c.cluster_name, COUNT(*) as cnt
        FROM role_clusters rc
        JOIN clusters c ON rc.cluster_id = c.cluster_id
        WHERE rc.is_primary = 1
        GROUP BY rc.cluster_id
        ORDER BY cnt DESC
    """).fetchall()
    print("\nJobs by primary cluster:")
    for name, cnt in rows:
        print(f"  {cnt:>4}  {name}")

    review_count = conn.execute(
        "SELECT COUNT(DISTINCT job_id) FROM role_clusters WHERE confidence = 'review' AND is_primary = 1"
    ).fetchone()[0]
    print(f"\nJobs flagged 'review' (primary): {review_count}")

    total = conn.execute("SELECT COUNT(DISTINCT job_id) FROM role_clusters").fetchone()[0]
    print(f"Total jobs mapped: {total}")

    print("\nSample mappings (mix of high + review):")
    samples = conn.execute("""
        SELECT rc.job_id, j.title, c.cluster_name, rc.is_primary, rc.confidence, rc.reasoning
        FROM role_clusters rc
        JOIN clusters c ON rc.cluster_id = c.cluster_id
        WHERE rc.is_primary = 1
        ORDER BY rc.confidence DESC, RANDOM()
        LIMIT 10
    """).fetchall()
    for jid, title, cluster, _, conf, reasoning in samples:
        print(f"  [{conf:>6}] {title[:35]:<35} → {cluster}")
        print(f"           {reasoning}")

    conn.close()


if __name__ == "__main__":
    main()
