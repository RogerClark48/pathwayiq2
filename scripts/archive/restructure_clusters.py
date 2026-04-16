"""
restructure_clusters.py
Implements the revised 19-cluster structure in job_roles_asset.db.

Steps:
  1. Back up current clusters and job cluster_id assignments
  2. Replace clusters table content (20 new clusters A1–O)
  3. Create job_cluster_secondary table
  4. Update jobs.cluster_id from normalized_title assignments
  5. Insert secondary memberships
  6. Verification report

Run from project root with venv active:
  venv/Scripts/python.exe scripts/restructure_clusters.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "job_roles_asset.db"

# ---------------------------------------------------------------------------
# Primary cluster assignments: normalized_title -> cluster_id
# ---------------------------------------------------------------------------
PRIMARY = {
    # A1 — Software development & testing
    "app developer":                     "A1",
    "applications developer":            "A1",
    "software developer":                "A1",
    "software engineer":                 "A1",
    "web developer":                     "A1",
    "computer games tester":             "A1",
    "software tester":                   "A1",
    "test lead":                         "A1",

    # A2 — Games, media & creative digital
    "computer games developer":          "A2",
    "game developer":                    "A2",
    "game artist":                       "A2",
    "game designer":                     "A2",
    "e-learning developer":              "A2",
    "multimedia programmer":             "A2",
    "multimedia specialist":             "A2",
    "ux designer":                       "A2",

    # B — AI & data science
    "artificial intelligence (ai) engineer": "B",
    "data analyst":                          "B",
    "data analyst-statistician":             "B",
    "data scientist":                        "B",
    "machine learning engineer":             "B",
    "operational researcher":                "B",
    "statistician":                          "B",

    # C1 — Cyber security
    "cyber intelligence officer":        "C1",
    "cyber security analyst":            "C1",
    "forensic computer analyst":         "C1",
    "it security co-ordinator":          "C1",
    "penetration tester":                "C1",

    # C2 — IT systems & infrastructure
    "application analyst":               "C2",
    "database administrator":            "C2",
    "it consultant":                     "C2",
    "it support technician":             "C2",
    "information systems manager":       "C2",
    "network engineer":                  "C2",
    "network manager":                   "C2",
    "solutions architect":               "C2",
    "systems analyst":                   "C2",
    "technical architect":               "C2",
    "it project manager":                "C2",
    "digital delivery manager":          "C2",

    # D1 — Building surveying & compliance
    "acoustic consultant":               "D1",
    "acoustics consultant":              "D1",
    "building control officer":          "D1",
    "building control surveyor":         "D1",
    "building surveyor":                 "D1",
    "fire safety engineer":              "D1",

    # D2 — Construction & civil engineering
    "architectural technician":          "D2",
    "architectural technologist":        "D2",
    "building site inspector":           "D2",
    "building technician":               "D2",
    "cad technician":                    "D2",
    "civil engineer":                    "D2",
    "civil engineering technician":      "D2",
    "construction contracts manager":    "D2",
    "construction manager":              "D2",
    "construction site supervisor":      "D2",
    "consulting civil engineer":         "D2",
    "contracting civil engineer":        "D2",
    "estimator":                         "D2",
    "quantity surveyor":                 "D2",
    "site engineer":                     "D2",
    "structural engineer":               "D2",

    # E1 — Maritime & naval
    "armed forces technical officer":    "E1",
    "marine engineer":                   "E1",
    "marine engineering technician":     "E1",
    "merchant navy deck officer":        "E1",
    "merchant navy engineering officer": "E1",
    "merchant navy officer":             "E1",
    "naval architect":                   "E1",

    # E2 — Aviation, space & weather
    "air accident investigator":         "E2",
    "air traffic controller":            "E2",
    "airline pilot":                     "E2",
    "astronaut":                         "E2",
    "astronomer":                        "E2",
    "climate scientist":                 "E2",
    "drone pilot":                       "E2",
    "meteorologist":                     "E2",
    "raf aviator":                       "E2",

    # F1 — Earth & geo sciences
    "drilling engineer":                 "F1",
    "engineering geologist":             "F1",
    "environmental engineer":            "F1",
    "geochemist":                        "F1",
    "geophysicist":                      "F1",
    "geoscientist":                      "F1",
    "geotechnical engineer":             "F1",
    "geotechnician":                     "F1",
    "hydrogeologist":                    "F1",
    "hydrologist":                       "F1",
    "minerals surveyor":                 "F1",
    "mining engineer":                   "F1",
    "petroleum engineer":                "F1",
    "quarry engineer":                   "F1",
    "seismologist":                      "F1",
    "water engineer":                    "F1",
    "water quality scientist":           "F1",

    # F2 — GIS, land surveying & agricultural engineering
    "agricultural engineer":             "F2",
    "agricultural engineering technician": "F2",
    "landbased engineer":                "F2",   # stored without hyphen
    "geographical information systems officer": "F2",
    "geospatial  technician":            "F2",   # two spaces — stored exactly
    "hydrographic surveyor":             "F2",
    "land surveyor":                     "F2",
    "landgeomatics surveyor":            "F2",   # stored without space/slash
    "surveying technician":              "F2",
    "field trials officer":              "F2",
    "plant breedergeneticist":           "F2",   # stored without slash

    # G — Life sciences & biological sciences
    "analytical chemist":                "G",
    "biochemist":                        "G",
    "biotechnologist":                   "G",
    "chemist":                           "G",
    "clinical scientist":                "G",
    "clinical scientist genomics":       "G",
    "geneticist":                        "G",
    "laboratory technician":             "G",
    "medicinal chemist":                 "G",
    "microbiologist":                    "G",
    "pharmacologist":                    "G",
    "research scientist":                "G",
    "scientific laboratory technician":  "G",
    "teaching laboratory technician":    "G",
    "food technologist":                 "G",

    # H — Materials science & physics
    "materials engineer":                "H",
    "materials technician":              "H",
    "nanotechnologist":                  "H",
    "physicist":                         "H",
    "metallurgist":                      "H",
    "metrologist":                       "H",
    "3d printing technician":            "H",

    # I — Precision & mechanical manufacturing
    "cnc machinist":                     "I",
    "engineering operative":             "I",
    "foundry moulder":                   "I",
    "non-destructive testing technician": "I",
    "toolmaker":                         "I",
    "welder":                            "I",
    "mechanical engineer":               "I",
    "mechanical engineering technician": "I",
    "design engineer":                   "I",
    "design and development engineer":   "I",
    "engineering construction technician": "I",
    "steel erector":                     "I",

    # J — Aerospace, automotive & robotics
    "aerospace engineer":                "J",
    "aerospace engineering technician":  "J",
    "automotive engineer":               "J",
    "car manufacturing worker":          "J",
    "motorsport engineer":               "J",
    "product designer":                  "J",
    "robotics engineer":                 "J",

    # K — Electrical, electronics & telecoms
    "auto electrician":                  "K",
    "broadcast engineer":                "K",
    "communications engineer":           "K",
    "electrical engineer":               "K",
    "electrical engineering technician": "K",
    "electrician":                       "K",
    "electronics engineer":              "K",
    "electronics engineering technician": "K",
    "telecoms engineer":                 "K",
    "technical sales engineer":          "K",
    "electricity distribution worker":   "K",
    "electricity generation worker":     "K",
    "rolling stock engineering technician": "K",
    "signalling technician":             "K",

    # L — Energy & renewables
    "building services engineer":        "L",
    "commercial energy assessor":        "L",
    "energy engineer":                   "L",
    "energy manager":                    "L",
    "heat pump engineer":                "L",
    "heating and ventilation engineer":  "L",
    "refrigeration and air-conditioning installer": "L",
    "renewable energy engineer":         "L",
    "thermal insulation engineer":       "L",

    # M — Plant & maintenance engineering
    "construction plant mechanic":       "M",
    "construction plant operator":       "M",
    "engineering maintenance technician": "M",
    "helicopter engineer":               "M",
    "lift engineer":                     "M",
    "maintenance engineer":              "M",
    "maintenance fitter":                "M",
    "pipe fitter":                       "M",
    "wind turbine technician":           "M",

    # N — Chemical & process engineering
    "chemical engineer":                 "N",
    "chemical engineering technician":   "N",
    "chemical plant process operator":   "N",
    "control and instrumentation engineer": "N",
    "manufacturing engineer":            "N",
    "manufacturing systems engineer":    "N",
    "packaging technologist":            "N",
    "productprocess development scientist": "N",   # stored without slash

    # O — Medical physics & clinical
    "biomedical engineer":               "O",
    "clinical engineer":                 "O",
    "clinical scientist medical physics": "O",
    "clinical technologist":             "O",
    "medical physicist":                 "O",
    "nuclear engineer":                  "O",
    "nuclear technician":                "O",
    "prosthetist and orthotist":         "O",
    "radiation protection practitioner": "O",
    "medical illustrator":               "O",
}


def main():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    # -----------------------------------------------------------------------
    # Pre-flight: check every title in PRIMARY exists in the DB
    # -----------------------------------------------------------------------
    print("Pre-flight checks...")
    zero_matches = []
    for nt in sorted(PRIMARY):
        n = db.execute("SELECT COUNT(*) FROM jobs WHERE normalized_title = ?", (nt,)).fetchone()[0]
        if n == 0:
            zero_matches.append(nt)

    if zero_matches:
        print(f"\nWARNING — {len(zero_matches)} title(s) in assignment list with no DB match:")
        for nt in zero_matches:
            print(f"  {repr(nt)}")
        print("\nAborting — fix the assignment list before proceeding.")
        db.close()
        return

    # Jobs with overview that are not in the assignment list
    all_nt = db.execute(
        "SELECT DISTINCT normalized_title, cluster_id FROM jobs "
        "WHERE overview IS NOT NULL AND cluster_id IS NOT NULL"
    ).fetchall()
    unassigned = [(r["normalized_title"], r["cluster_id"]) for r in all_nt
                  if r["normalized_title"] not in PRIMARY]
    if unassigned:
        print(f"NOTE — {len(unassigned)} normalized_title(s) with overview not in assignment list "
              f"(cluster_id will be preserved as-is):")
        for nt, cid in unassigned:
            print(f"  current cluster={cid}  {repr(nt)}")

    print("Pre-flight OK — proceeding.\n")

    # -----------------------------------------------------------------------
    # All changes in one transaction
    # -----------------------------------------------------------------------
    with db:
        # -------------------------------------------------------------------
        # Step 1 — Backups
        # -------------------------------------------------------------------
        print("Step 1 — Creating backups...")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS clusters_backup_k21
                AS SELECT * FROM clusters;
            CREATE TABLE IF NOT EXISTS jobs_cluster_backup_k21
                AS SELECT id, cluster_id FROM jobs;
        """)
        n_backup = db.execute("SELECT COUNT(*) FROM clusters_backup_k21").fetchone()[0]
        print(f"  clusters_backup_k21: {n_backup} rows")
        n_job_backup = db.execute("SELECT COUNT(*) FROM jobs_cluster_backup_k21").fetchone()[0]
        print(f"  jobs_cluster_backup_k21: {n_job_backup} rows")

        # -------------------------------------------------------------------
        # Step 2 — Replace clusters table content
        # -------------------------------------------------------------------
        print("\nStep 2 — Replacing clusters table...")
        db.execute("DELETE FROM clusters")
        db.executemany(
            "INSERT INTO clusters (cluster_id, name) VALUES (?, ?)",
            [
                ("A1", "Software development & testing"),
                ("A2", "Games, media & creative digital"),
                ("B",  "AI & data science"),
                ("C1", "Cyber security"),
                ("C2", "IT systems & infrastructure"),
                ("D1", "Building surveying & compliance"),
                ("D2", "Construction & civil engineering"),
                ("E1", "Maritime & naval"),
                ("E2", "Aviation, space & weather"),
                ("F1", "Earth & geo sciences"),
                ("F2", "GIS, land surveying & agricultural engineering"),
                ("G",  "Life sciences & biological sciences"),
                ("H",  "Materials science & physics"),
                ("I",  "Precision & mechanical manufacturing"),
                ("J",  "Aerospace, automotive & robotics"),
                ("K",  "Electrical, electronics & telecoms"),
                ("L",  "Energy & renewables"),
                ("M",  "Plant & maintenance engineering"),
                ("N",  "Chemical & process engineering"),
                ("O",  "Medical physics & clinical"),
            ]
        )
        n_clusters = db.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        print(f"  Inserted {n_clusters} clusters")

        # -------------------------------------------------------------------
        # Step 3 — Create secondary membership table
        # -------------------------------------------------------------------
        print("\nStep 3 — Creating job_cluster_secondary table...")
        db.execute("""
            CREATE TABLE IF NOT EXISTS job_cluster_secondary (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id     INTEGER,
                cluster_id TEXT,
                note       TEXT
            )
        """)
        db.execute("DELETE FROM job_cluster_secondary")
        print("  Table ready (existing rows cleared)")

        # -------------------------------------------------------------------
        # Step 4 — Primary cluster assignments
        # -------------------------------------------------------------------
        print("\nStep 4 — Updating primary cluster assignments...")
        total_updated = 0
        for nt, cid in PRIMARY.items():
            cursor = db.execute(
                "UPDATE jobs SET cluster_id = ? WHERE normalized_title = ?",
                (cid, nt)
            )
            total_updated += cursor.rowcount
        print(f"  {total_updated} job rows updated")

        # -------------------------------------------------------------------
        # Step 5 — Secondary memberships
        # -------------------------------------------------------------------
        print("\nStep 5 — Inserting secondary memberships...")

        secondaries = [
            # medical illustrator: primary O, secondary A2
            ("medical illustrator",         "A2", "Creative digital skills — secondary membership"),
            # technical author: secondary C2, D2, N
            ("technical author",            "C2", "Software/IT documentation context"),
            ("technical author",            "D2", "Engineering documentation context"),
            ("technical author",            "N",  "Chemical/process documentation context"),
            # field trials officer: primary F2, secondary G
            ("field trials officer",        "G",  "Applied biological sciences context"),
            # plant breeder/geneticist: primary F2, secondary G
            ("plant breedergeneticist",     "G",  "Biological sciences professional community"),
            # physicist: primary H, secondary G
            ("physicist",                   "G",  "Physics underpins life sciences research"),
            # analytical chemist: primary G, secondary H
            ("analytical chemist",          "H",  "Materials and analytical chemistry overlap"),
            # chemist: primary G, secondary H
            ("chemist",                     "H",  "Materials and chemistry overlap"),
            # food technologist: primary G, secondary N
            ("food technologist",           "N",  "Food production and process engineering context"),
            # mechanical engineer: primary I, secondary J
            ("mechanical engineer",         "J",  "Gateway into aerospace and automotive engineering"),
            # mechanical engineering technician: primary I, secondary J
            ("mechanical engineering technician", "J", "Technician route into aerospace/automotive"),
            # design engineer: primary I, secondary J
            ("design engineer",             "J",  "Design engineering spans manufacturing and aerospace"),
            # design and development engineer: primary I, secondary J
            ("design and development engineer", "J", "Development engineering spans manufacturing and aerospace"),
            # engineering construction technician: primary I, secondary J
            ("engineering construction technician", "J", "Construction technician work overlaps aerospace/automotive"),
            # car manufacturing worker: primary J, secondary I
            ("car manufacturing worker",    "I",  "Manufacturing floor context"),
            # robotics engineer: primary J, secondary B and K
            ("robotics engineer",           "B",  "AI and machine learning systems overlap"),
            ("robotics engineer",           "K",  "Electrical and electronics systems overlap"),
            # helicopter engineer: primary M, secondary E2
            ("helicopter engineer",         "E2", "Aviation industry context"),
            # control and instrumentation engineer: primary N, secondary K
            ("control and instrumentation engineer", "K", "Electrical and control systems overlap"),
            # it project manager: primary C2, secondary A2
            ("it project manager",          "A2", "Digital project management context"),
            # digital delivery manager: primary C2, secondary A2
            ("digital delivery manager",    "A2", "Digital delivery context"),
        ]

        # production manager% → secondary I and N (matches both normalized_titles)
        prod_mgr_secondaries = [
            ("I", "Manufacturing management context"),
            ("N", "Chemical/process production management context"),
        ]

        sec_count = 0
        for nt, cid, note in secondaries:
            rows = db.execute(
                "SELECT id FROM jobs WHERE normalized_title = ?", (nt,)
            ).fetchall()
            for row in rows:
                db.execute(
                    "INSERT INTO job_cluster_secondary (job_id, cluster_id, note) VALUES (?,?,?)",
                    (row["id"], cid, note)
                )
                sec_count += 1

        for cid, note in prod_mgr_secondaries:
            rows = db.execute(
                "SELECT id FROM jobs WHERE normalized_title LIKE 'production manager%'"
            ).fetchall()
            for row in rows:
                db.execute(
                    "INSERT INTO job_cluster_secondary (job_id, cluster_id, note) VALUES (?,?,?)",
                    (row["id"], cid, note)
                )
                sec_count += 1

        print(f"  {sec_count} secondary membership rows inserted")

    # -----------------------------------------------------------------------
    # Step 6 — Verification
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Jobs with overview and no cluster assignment
    n_unassigned = db.execute(
        "SELECT COUNT(*) FROM jobs WHERE cluster_id IS NULL AND overview IS NOT NULL"
    ).fetchone()[0]
    print(f"\nJobs with overview and NULL cluster_id: {n_unassigned}")

    # Cluster population counts
    print("\nCluster populations:")
    rows = db.execute("""
        SELECT c.cluster_id, c.name, COUNT(j.id) as job_count
        FROM clusters c
        LEFT JOIN jobs j ON j.cluster_id = c.cluster_id
        GROUP BY c.cluster_id
        ORDER BY c.cluster_id
    """).fetchall()
    for r in rows:
        print(f"  {r['cluster_id']:>4}  {r['name']:<45}  {r['job_count']:>3} jobs")

    # Secondary membership counts
    print("\nSecondary memberships per cluster:")
    rows = db.execute("""
        SELECT cluster_id, COUNT(*) as n
        FROM job_cluster_secondary
        GROUP BY cluster_id
        ORDER BY cluster_id
    """).fetchall()
    for r in rows:
        print(f"  {r['cluster_id']:>4}  {r['n']} secondary records")

    # Flag jobs whose cluster_id has no matching clusters row (orphan check)
    orphans = db.execute("""
        SELECT DISTINCT j.cluster_id FROM jobs j
        LEFT JOIN clusters c ON c.cluster_id = j.cluster_id
        WHERE j.cluster_id IS NOT NULL AND c.cluster_id IS NULL
    """).fetchall()
    if orphans:
        print(f"\nWARNING — orphaned cluster_id values (not in clusters table):")
        for r in orphans:
            print(f"  {r['cluster_id']}")
    else:
        print("\nNo orphaned cluster_id values.")

    # Spot-check first 50 assignments
    print("\nSpot-check — first 50 by cluster_id, normalized_title:")
    rows = db.execute("""
        SELECT normalized_title, cluster_id FROM jobs
        WHERE cluster_id IS NOT NULL
        ORDER BY cluster_id, normalized_title
        LIMIT 50
    """).fetchall()
    for r in rows:
        print(f"  {r['cluster_id']:>4}  {r['normalized_title']}")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
