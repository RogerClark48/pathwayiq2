"""
match_profile_diagnostic.py
---------------------------
Computes a four-dimension match profile for course-career pairs.

Dimensions:
  Domain  -- course _overview      vs job _overview       (same world?)
  Skills  -- course _learning      vs job _skills_only    (course teaches what job needs?)
  Duties  -- course _learning      vs job _duties         (course content vs daily work?)
  Pathway -- course _progression   vs job _entry          (course leads where job requires?)

Part 1: Specific pair analysis (genuine / hard case / spurious)
Part 2: Full corpus -- 83 courses x top-10 careers (830 profiles)

All vectors preloaded into memory at startup.
No per-pair Chroma calls -- avoids Rust HNSW reader instability.
Score formula: cosine_similarity * 100  (both collections use hnsw:space=cosine)
"""

import sqlite3
import statistics
import numpy as np
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_PATH = r"C:\Dev\pathwayiq\chroma_store"
GMIOT_DB    = r"C:\Dev\pathwayiq\gmiot.sqlite"
JOBS_DB     = r"C:\Dev\pathwayiq\emiot_jobs_asset.db"
TOP_N       = 10

# ---------------------------------------------------------------------------
# DB connections
# ---------------------------------------------------------------------------
gmiot_conn = sqlite3.connect(GMIOT_DB)
gmiot_conn.row_factory = sqlite3.Row

jobs_conn = sqlite3.connect(JOBS_DB)
jobs_conn.row_factory = sqlite3.Row


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------
def load_collection(client, name: str) -> dict[str, np.ndarray]:
    """Bulk-load all embeddings from a collection into {chunk_id: ndarray}."""
    col    = client.get_collection(name)
    result = col.get(include=["embeddings"])
    return {
        chunk_id: np.array(emb, dtype=np.float32)
        for chunk_id, emb in zip(result["ids"], result["embeddings"])
    }


print("Loading vectors from Chroma (7 collections)...", flush=True)
_chroma = chromadb.PersistentClient(path=CHROMA_PATH)

COURSE_OVERVIEW    = load_collection(_chroma, "gmiot_courses")          # {id}_overview + {id}_skills
COURSE_LEARNING    = load_collection(_chroma, "gmiot_courses_learning") # {id}_learning
COURSE_PROGRESSION = load_collection(_chroma, "gmiot_courses_progression") # {id}_progression
JOB_OVERVIEW       = load_collection(_chroma, "gmiot_jobs")             # {id}_overview + {id}_skills
JOB_SKILLS         = load_collection(_chroma, "gmiot_jobs_skills")      # {id}_skills_only
JOB_DUTIES         = load_collection(_chroma, "gmiot_jobs_duties")      # {id}_duties
JOB_ENTRY          = load_collection(_chroma, "gmiot_jobs_entry")       # {id}_entry

print(f"  gmiot_courses:            {len(COURSE_OVERVIEW)} chunks")
print(f"  gmiot_courses_learning:   {len(COURSE_LEARNING)} chunks")
print(f"  gmiot_courses_progression:{len(COURSE_PROGRESSION)} chunks")
print(f"  gmiot_jobs:               {len(JOB_OVERVIEW)} chunks")
print(f"  gmiot_jobs_skills:        {len(JOB_SKILLS)} chunks")
print(f"  gmiot_jobs_duties:        {len(JOB_DUTIES)} chunks")
print(f"  gmiot_jobs_entry:         {len(JOB_ENTRY)} chunks")
print("Done.\n", flush=True)

# Build job overview matrix for fast top-N search in Part 2
# Filter JOB_OVERVIEW to only _overview chunks (collection also has _skills chunks)
_job_overview_ids  = sorted(
    [k for k in JOB_OVERVIEW if k.endswith("_overview")],
    key=lambda k: int(k.split("_")[0])
)
_job_matrix        = np.stack([JOB_OVERVIEW[k] for k in _job_overview_ids])  # (N, 1024)
_job_matrix_norms  = np.linalg.norm(_job_matrix, axis=1, keepdims=True)
_job_matrix_normed = _job_matrix / np.maximum(_job_matrix_norms, 1e-9)       # pre-normalised


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cosine_score(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> int | None:
    if vec_a is None or vec_b is None:
        return None
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return None
    return round(float(np.dot(vec_a, vec_b) / (norm_a * norm_b)) * 100)


def top_n_jobs(course_id: int, n: int = TOP_N) -> list[dict]:
    """Return top-N job matches by domain (overview vs overview) using matrix multiply."""
    ov = COURSE_OVERVIEW.get(f"{course_id}_overview")
    if ov is None:
        return []
    norm = np.linalg.norm(ov)
    if norm == 0:
        return []
    sims = _job_matrix_normed @ (ov / norm)      # (N,)
    idx  = np.argsort(sims)[::-1][:n]
    results = []
    for i in idx:
        chunk_id = _job_overview_ids[i]
        job_id   = chunk_id.replace("_overview", "")
        results.append({"job_id": job_id, "domain_score": round(float(sims[i]) * 100)})
    return results


def compute_profile(course_id: int | str, job_id: int | str) -> dict:
    cid = str(course_id)
    jid = str(job_id)

    ov = COURSE_OVERVIEW.get(f"{cid}_overview")
    lv = COURSE_LEARNING.get(f"{cid}_learning")
    pv = COURSE_PROGRESSION.get(f"{cid}_progression")

    jo = JOB_OVERVIEW.get(f"{jid}_overview")
    js = JOB_SKILLS.get(f"{jid}_skills_only")
    jd = JOB_DUTIES.get(f"{jid}_duties")
    je = JOB_ENTRY.get(f"{jid}_entry")

    return {
        "domain":  cosine_score(ov, jo),
        "skills":  cosine_score(lv, js),
        "duties":  cosine_score(lv, jd),
        "pathway": cosine_score(pv, je),
    }


def classify_profile(p: dict) -> str:
    scores = [v for v in p.values() if v is not None]
    if not scores:
        return "INSUFFICIENT DATA"
    mean_s = statistics.mean(scores)
    if mean_s < 55:
        return "WEAK"
    if all(s >= 70 for s in scores):
        return "STRONG"
    d = p.get("domain") or 0
    low_others = sum(1 for k in ("skills", "duties", "pathway") if p.get(k) is not None and p[k] < 65)
    if d > 70 and low_others >= 2:
        return "MIXED"
    if max(scores) - min(scores) <= 15:
        return "CONSISTENT"
    return "MIXED"


def bar(score: int | None, width: int = 10) -> str:
    if score is None:
        return "." * width
    filled = max(0, min(width, round(score / (100 / width))))
    return "#" * filled + "." * (width - filled)


SSA_SHORT = {
    "Engineering and Manufacturing Technologies":       "Engineering",
    "Construction, Planning and the Built Environment": "Construction",
    "Information and Communication Technology":         "ICT",
    "Health, Public Services and Care":                 "Health",
    "Arts, Media and Publishing":                       "Arts & Media",
    "Social Sciences":                                  "Social Sciences",
    "Education and Training":                           "Education",
    "Business, Administration and Law":                 "Business",
    "Retail and Commercial Enterprise":                 "Retail",
    "Preparation for Life and Work":                    "Prep for Work",
}

# ---------------------------------------------------------------------------
# DB lookups
# ---------------------------------------------------------------------------
def find_course(fragment: str) -> dict | None:
    row = gmiot_conn.execute(
        "SELECT course_id, course_title, ssa_label FROM gmiot_courses "
        "WHERE course_title LIKE ? ORDER BY course_id LIMIT 1",
        (f"%{fragment}%",)
    ).fetchone()
    return dict(row) if row else None


def find_job(fragment: str) -> dict | None:
    row = jobs_conn.execute(
        "SELECT id, title, source FROM jobs WHERE title LIKE ? ORDER BY id LIMIT 1",
        (f"%{fragment}%",)
    ).fetchone()
    return dict(row) if row else None


def job_title(job_id: str) -> str:
    row = jobs_conn.execute("SELECT title FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return row["title"] if row else f"job {job_id}"


# ---------------------------------------------------------------------------
# Part 1 — Specific pair analysis
# ---------------------------------------------------------------------------
TEST_PAIRS = [
    # (course_fragment, job_fragment, label)
    ("Nursing Associate FdSc",                    "Adult nurse",                 "GENUINE"),
    ("Digital Software Development",              "Software developer",           "GENUINE"),
    ("Construction Management (Site Supervisor)", "Construction site supervisor", "GENUINE"),
    ("HNC Construction and Civil Engineering",    "Civil engineer",               "GENUINE"),
    ("Nursing Associate FdSc",                    "Veterinary nurse",             "HARD CASE"),
    ("Nursing Associate FdSc",                    "Animal physiotherapist",       "HARD CASE"),
    ("Supporting the Adult Nursing Team",         "Veterinary nurse",             "HARD CASE"),
    ("Healthcare Practice",                       "Physiotherapist",              "HARD CASE"),
    ("Construction Management (Site Supervisor)", "Software developer",           "SPURIOUS"),
    ("Digital Software Development",              "Construction site supervisor",  "SPURIOUS"),
    ("Nursing Associate FdSc",                    "Mechanical engineer",          "SPURIOUS"),
    ("HNC Construction and Civil Engineering",    "Adult nurse",                  "SPURIOUS"),
]


def print_profile_block(label: str, course_title: str, job_title_: str, p: dict) -> None:
    def fmt(v): return f"{v:>3}%" if v is not None else " N/A"
    print("=" * 70)
    print(f"{label}: {course_title[:35]} -> {job_title_[:28]}")
    print("=" * 70)
    print(f"  Domain  (overview vs overview):   {fmt(p['domain'])}  {bar(p['domain'])}")
    print(f"  Skills  (learning vs skills):      {fmt(p['skills'])}  {bar(p['skills'])}")
    print(f"  Duties  (learning vs duties):      {fmt(p['duties'])}  {bar(p['duties'])}")
    print(f"  Pathway (progression vs entry):    {fmt(p['pathway'])}  {bar(p['pathway'])}")
    print(f"  -> {classify_profile(p)}")
    print()


def run_part1() -> None:
    print("\n" + "=" * 70)
    print("PART 1 -- SPECIFIC PAIR ANALYSIS")
    print("=" * 70 + "\n")

    rows = []
    for course_frag, job_frag, label in TEST_PAIRS:
        course = find_course(course_frag)
        job    = find_job(job_frag)
        if course is None:
            print(f"[SKIP] Course not found: {course_frag!r}")
            continue
        if job is None:
            print(f"[SKIP] Job not found: {job_frag!r}")
            continue
        p = compute_profile(course["course_id"], job["id"])
        print_profile_block(label, course["course_title"], job["title"], p)
        rows.append((label, course["course_title"], job["title"], p))

    # Summary table
    W = 97
    print("=" * W)
    print(f"  {'Label':<10}  {'Course':<33}  {'Job':<27}  Domain  Skills  Duties  Pathway  Class")
    print("-" * W)
    for label, ct, jt, p in rows:
        def s(v): return f"{v:>3}%" if v is not None else " N/A"
        cls = classify_profile(p)
        print(f"  {label:<10}  {ct[:33]:<33}  {jt[:27]:<27}  "
              f"{s(p['domain'])}    {s(p['skills'])}   {s(p['duties'])}    {s(p['pathway'])}   {cls}")
    print("=" * W)


# ---------------------------------------------------------------------------
# Part 2 — Full corpus analysis
# ---------------------------------------------------------------------------
def run_part2() -> None:
    print("\n\n" + "=" * 70)
    print("PART 2 -- FULL CORPUS ANALYSIS")
    print("=" * 70 + "\n")

    courses = gmiot_conn.execute(
        "SELECT course_id, course_title, ssa_label FROM gmiot_courses ORDER BY course_id"
    ).fetchall()

    domain_scores  = []
    skills_scores  = []
    duties_scores  = []
    pathway_scores = []
    class_counts   = {"STRONG": 0, "CONSISTENT": 0, "MIXED": 0, "WEAK": 0}
    dom_skills_div = []   # (delta, course_title, job_title, domain, skills)
    pathway_low    = []   # (pathway, domain, course_title, job_title)
    ssa_scores     = {}   # ssa -> {domain:[], skills:[], duties:[], pathway:[]}
    skipped        = 0

    for course in courses:
        cid   = course["course_id"]
        ct    = course["course_title"]
        ssa   = course["ssa_label"] or "Unknown"

        matches = top_n_jobs(cid, TOP_N)
        if not matches:
            skipped += 1
            continue

        if ssa not in ssa_scores:
            ssa_scores[ssa] = {"domain": [], "skills": [], "duties": [], "pathway": []}

        for m in matches:
            jid = m["job_id"]
            p   = compute_profile(cid, jid)
            jt  = job_title(jid)

            domain_scores.append(p["domain"])
            if p["skills"]  is not None: skills_scores.append(p["skills"])
            if p["duties"]  is not None: duties_scores.append(p["duties"])
            if p["pathway"] is not None: pathway_scores.append(p["pathway"])

            ssa_scores[ssa]["domain"].append(p["domain"])
            if p["skills"]  is not None: ssa_scores[ssa]["skills"].append(p["skills"])
            if p["duties"]  is not None: ssa_scores[ssa]["duties"].append(p["duties"])
            if p["pathway"] is not None: ssa_scores[ssa]["pathway"].append(p["pathway"])

            cls = classify_profile(p)
            class_counts[cls] = class_counts.get(cls, 0) + 1

            if p["skills"] is not None:
                delta = p["domain"] - p["skills"]
                dom_skills_div.append((delta, ct, jt, p["domain"], p["skills"]))

            if p["pathway"] is not None:
                pathway_low.append((p["pathway"], p["domain"], ct, jt))

    total     = len(domain_scores)
    total_cls = sum(class_counts.values())

    def dl(label, scores):
        if not scores: return f"  {label:<8} no data"
        return (f"  {label:<8} mean={statistics.mean(scores):.1f}%  "
                f"median={statistics.median(scores):.0f}%  "
                f"min={min(scores)}%  max={max(scores)}%  "
                f"std={statistics.stdev(scores):.1f}%  "
                f"range={max(scores)-min(scores)}%")

    print(f"{len(courses)-skipped} courses, top-{TOP_N} matches, {total} pairs\n")

    print("Dimension score distributions:")
    print(dl("Domain:", domain_scores))
    print(dl("Skills:", skills_scores))
    print(dl("Duties:", duties_scores))
    print(dl("Pathway:", pathway_scores))

    print("\nDiscriminating power (std dev):")
    for label, sc in [("Domain", domain_scores), ("Skills", skills_scores),
                       ("Duties", duties_scores), ("Pathway", pathway_scores)]:
        if sc:
            print(f"  {label:<8} std={statistics.stdev(sc):.1f}%  range={max(sc)-min(sc)}%")

    print(f"\nProfile classifications ({total_cls} pairs):")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<12} {count:>4}  ({count/total_cls*100:.0f}%)")

    dom_skills_div.sort(key=lambda x: x[0], reverse=True)
    print("\nLargest domain > skills divergences (possibly spurious connections):")
    for delta, ct, jt, dom, sk in dom_skills_div[:10]:
        print(f"  {ct[:30]:<30} -> {jt[:30]:<30}  domain={dom}% skills={sk}% d={delta:+d}%")

    pathway_low.sort(key=lambda x: x[0])
    print("\nPairs where pathway scores lowest (most discriminating):")
    for pw, dom, ct, jt in pathway_low[:10]:
        print(f"  {ct[:30]:<30} -> {jt[:30]:<30}  pathway={pw}% domain={dom}%")

    print("\nMean profile by subject area:")
    print(f"  {'SSA':<20}  Domain  Skills  Duties  Pathway  Pairs")
    print("-" * 65)
    def m(lst): return f"{statistics.mean(lst):.0f}%" if lst else " N/A"
    for ssa in sorted(ssa_scores):
        d = ssa_scores[ssa]
        short = SSA_SHORT.get(ssa, ssa[:15])
        print(f"  {short:<20}  {m(d['domain']):<6}  {m(d['skills']):<6}  "
              f"{m(d['duties']):<6}  {m(d['pathway']):<7}  {len(d['domain'])}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_part1()
    run_part2()
