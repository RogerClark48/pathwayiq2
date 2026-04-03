"""
check_connections.py — Readable report of approved job connections for a sample of courses.
"""

import os
import sqlite3

ROOT           = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CONNECTIONS_DB = os.path.join(ROOT, "connections.db")
GMIOT_DB       = os.path.join(ROOT, "gmiot.sqlite")
JOBS_DB        = os.path.join(ROOT, "job_roles_asset.db")

# ---------------------------------------------------------------------------
# Courses to inspect — (partial title match, reason)
# ---------------------------------------------------------------------------
TARGETS = [
    # Previously-zero courses
    ("Access to Higher Education Diploma – Social Science",    "prev-zero"),
    ("Access to HE Diploma (Computing and Digital Technology)","prev-zero"),
    ("City & Guilds Level 4 Award",                            "prev-zero"),
    ("DipHE Esports Enterprise and Management",                "prev-zero"),
    # Sample across subject areas
    ("Nursing Associate FdSc",                                 "nursing"),
    ("HNC Mechanical Engineering for England",                 "engineering"),
    ("HNC Construction Management for England (Site Superviso","construction"),
    ("HND Computing for England (Cyber Security)",             "digital"),
    ("BA (Hons) Creative Practitioner",                        "creative"),
]

gconn = sqlite3.connect(GMIOT_DB)
gconn.row_factory = sqlite3.Row
cconn = sqlite3.connect(CONNECTIONS_DB)
cconn.row_factory = sqlite3.Row
cconn.execute(f"ATTACH DATABASE ? AS jobs_db", (JOBS_DB,))

for title_fragment, label in TARGETS:
    course = gconn.execute(
        "SELECT course_id, course_title, qual_type, level FROM gmiot_courses "
        "WHERE course_title LIKE ? LIMIT 1",
        (f"%{title_fragment}%",)
    ).fetchone()

    if not course:
        print(f"[{label}] *** NOT FOUND: {title_fragment!r} ***\n")
        continue

    cid = course["course_id"]
    rows = cconn.execute(
        """SELECT c.job_id, c.semantic_score, c.skills_score, j.title, j.level as job_level
           FROM course_job_connections c
           JOIN jobs_db.jobs j ON j.id = c.job_id
           WHERE c.course_id = ?
           ORDER BY c.semantic_score DESC""",
        (cid,)
    ).fetchall()

    print("-" * 70)
    print(f"[{label}] {course['course_title']}")
    print(f"  {course['qual_type'] or '—'}  |  Course Level {course['level'] or '?'} (max job level allowed: {course['level'] + 2 if course['level'] else '—'})  |  {len(rows)} connections")
    print()
    if rows:
        for r in rows:
            sk  = f"  skills={r['skills_score']}%" if r["skills_score"] is not None else ""
            lvl = f"  L{r['job_level']}" if r["job_level"] is not None else "  L?"
            print(f"  {r['semantic_score']:>3}%{sk:<14}{lvl:<6}  {r['title']}")
    else:
        print("  *** NO CONNECTIONS ***")
    print()

gconn.close()
cconn.close()
