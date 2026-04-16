"""
promote_cluster_pathways.py
Part 1: Apply staging fixes (cluster 13 merges, cluster 11 meteorologist check).
Part 2: Duplication check (review log only).
Part 3/4: Promote staging tables from job_roles_asset.db to production tables in gmiot.sqlite.
"""

import sqlite3
import sys

JOBS_DB  = r"C:\Dev\pathwayiq2\job_roles_asset.db"
GMIOT_DB = r"C:\Dev\pathwayiq2\gmiot.sqlite"


# ── Part 1 — Staging fixes ────────────────────────────────────────────────────

def apply_staging_fixes(db):
    print("=" * 60)
    print("PART 1 — STAGING FIXES")
    print("=" * 60)

    # ── Fix 1: Cluster 13 — merge "Physics and nanotechnology" into "Chemistry and materials science" ──

    src = db.execute(
        "SELECT id FROM track_staging WHERE cluster_id='13' AND track_name LIKE '%Physics%nanotechnology%'"
    ).fetchone()
    dst = db.execute(
        "SELECT id FROM track_staging WHERE cluster_id='13' AND track_name LIKE '%Chemistry%materials%'"
    ).fetchone()

    if not src or not dst:
        print(f"Fix 1: SKIPPED — could not find source ({src}) or destination ({dst}) track")
    else:
        src_id, dst_id = src["id"], dst["id"]
        # Only move jobs not already in the destination track
        existing = {r["job_id"] for r in db.execute(
            "SELECT job_id FROM track_jobs_staging WHERE track_staging_id=?", (dst_id,)
        )}
        rows = db.execute(
            "SELECT job_id, title, job_order FROM track_jobs_staging WHERE track_staging_id=?", (src_id,)
        ).fetchall()
        moved = 0
        max_order = db.execute(
            "SELECT COALESCE(MAX(job_order),0) FROM track_jobs_staging WHERE track_staging_id=?", (dst_id,)
        ).fetchone()[0]
        for row in rows:
            if row["job_id"] not in existing:
                max_order += 1
                db.execute(
                    "INSERT INTO track_jobs_staging (track_staging_id, job_id, title, job_order) VALUES (?,?,?,?)",
                    (dst_id, row["job_id"], row["title"], max_order),
                )
                moved += 1
            else:
                print(f"  Fix 1: job {row['job_id']} ({row['title']}) already in destination — skipped")
        db.execute("DELETE FROM track_jobs_staging WHERE track_staging_id=?", (src_id,))
        db.execute("DELETE FROM track_staging WHERE id=?", (src_id,))
        print(f"Fix 1: merged 'Physics and nanotechnology' (id={src_id}) into 'Chemistry and materials science' (id={dst_id}) — {moved} jobs moved")

    # ── Fix 2: Cluster 13 — merge "Pharmaceutical sciences" into "Biological sciences and research" ──

    src2 = db.execute(
        "SELECT id FROM track_staging WHERE cluster_id='13' AND track_name LIKE '%Pharmaceutical%'"
    ).fetchone()
    dst2 = db.execute(
        "SELECT id FROM track_staging WHERE cluster_id='13' AND track_name LIKE '%Biological%'"
    ).fetchone()

    if not src2 or not dst2:
        print(f"Fix 2: SKIPPED — could not find source ({src2}) or destination ({dst2}) track")
    else:
        src2_id, dst2_id = src2["id"], dst2["id"]
        existing2 = {r["job_id"] for r in db.execute(
            "SELECT job_id FROM track_jobs_staging WHERE track_staging_id=?", (dst2_id,)
        )}
        rows2 = db.execute(
            "SELECT job_id, title, job_order FROM track_jobs_staging WHERE track_staging_id=?", (src2_id,)
        ).fetchall()
        moved2 = 0
        max_order2 = db.execute(
            "SELECT COALESCE(MAX(job_order),0) FROM track_jobs_staging WHERE track_staging_id=?", (dst2_id,)
        ).fetchone()[0]
        for row in rows2:
            if row["job_id"] not in existing2:
                max_order2 += 1
                db.execute(
                    "INSERT INTO track_jobs_staging (track_staging_id, job_id, title, job_order) VALUES (?,?,?,?)",
                    (dst2_id, row["job_id"], row["title"], max_order2),
                )
                moved2 += 1
            else:
                print(f"  Fix 2: job {row['job_id']} ({row['title']}) already in destination — skipped")
        db.execute("DELETE FROM track_jobs_staging WHERE track_staging_id=?", (src2_id,))
        db.execute("DELETE FROM track_staging WHERE id=?", (src2_id,))
        print(f"Fix 2: merged 'Pharmaceutical sciences' (id={src2_id}) into 'Biological sciences and research' (id={dst2_id}) — {moved2} jobs moved")

    # ── Fix 3: Cluster 11 — remove meteorologist (job 443) from Track 1 if present ──

    atc_track = db.execute(
        "SELECT id FROM track_staging WHERE cluster_id='11' AND track_name LIKE '%Air traffic control%'"
    ).fetchone()
    if not atc_track:
        print("Fix 3: SKIPPED — could not find 'Air traffic control' track for cluster 11")
    else:
        atc_id = atc_track["id"]
        row = db.execute(
            "SELECT id FROM track_jobs_staging WHERE track_staging_id=? AND job_id=443", (atc_id,)
        ).fetchone()
        if row:
            db.execute("DELETE FROM track_jobs_staging WHERE id=?", (row["id"],))
            print(f"Fix 3: removed meteorologist (job 443) from 'Air traffic control' track (id={atc_id})")
        else:
            print(f"Fix 3: meteorologist (job 443) not present in 'Air traffic control' track (id={atc_id}) — no action needed")

    # ── Re-number track_order for Cluster 13 ──

    surviving = db.execute(
        "SELECT id FROM track_staging WHERE cluster_id='13' ORDER BY track_order"
    ).fetchall()
    for new_order, row in enumerate(surviving, 1):
        db.execute("UPDATE track_staging SET track_order=? WHERE id=?", (new_order, row["id"]))
    print(f"Re-numbered Cluster 13 tracks: {len(surviving)} tracks now ordered 1–{len(surviving)}")

    db.commit()
    print()


# ── Part 2 — Duplication check ────────────────────────────────────────────────

def duplication_check(db):
    print("=" * 60)
    print("PART 2 — DUPLICATION CHECK")
    print("=" * 60)

    rows = db.execute("""
        SELECT ts.cluster_id, cps.cluster_name, j.normalized_title,
               COUNT(DISTINCT ts.id) as track_count,
               GROUP_CONCAT(ts.track_name, ' | ') as track_names
        FROM track_jobs_staging tjs
        JOIN track_staging ts ON ts.id = tjs.track_staging_id
        JOIN jobs j ON j.id = tjs.job_id
        JOIN cluster_pathway_staging cps ON cps.cluster_id = ts.cluster_id
        GROUP BY ts.cluster_id, j.normalized_title
        HAVING track_count > 1
        ORDER BY ts.cluster_id, j.normalized_title
    """).fetchall()

    if rows:
        print(f"Found {len(rows)} duplicated job title(s) across tracks:\n")
        for r in rows:
            print(f"  Cluster {r['cluster_id']} ({r['cluster_name']}): '{r['normalized_title']}' in {r['track_count']} tracks")
            print(f"    Tracks: {r['track_names']}")
    else:
        print("No duplications found.")
    print()


# ── Part 3 — Create production tables ─────────────────────────────────────────

def create_production_tables(gdb):
    gdb.executescript("""
        CREATE TABLE IF NOT EXISTS cluster_narratives (
            cluster_id        TEXT PRIMARY KEY,
            cluster_name      TEXT,
            cluster_narrative TEXT,
            generated_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS career_tracks (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id       TEXT,
            track_name       TEXT,
            track_narrative  TEXT,
            track_order      INTEGER
        );

        CREATE TABLE IF NOT EXISTS career_track_jobs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            career_track_id  INTEGER,
            job_id           INTEGER,
            title            TEXT,
            job_order        INTEGER,
            FOREIGN KEY (career_track_id) REFERENCES career_tracks(id)
        );
    """)
    gdb.commit()


# ── Part 4 — Promote ──────────────────────────────────────────────────────────

def promote(db, gdb):
    print("=" * 60)
    print("PART 3/4 — PROMOTION TO PRODUCTION")
    print("=" * 60)

    # Clear production tables
    gdb.execute("DELETE FROM career_track_jobs")
    gdb.execute("DELETE FROM career_tracks")
    gdb.execute("DELETE FROM cluster_narratives")
    gdb.commit()
    print("Production tables cleared.")

    # cluster_narratives
    clusters = db.execute(
        "SELECT cluster_id, cluster_name, cluster_narrative, generated_at FROM cluster_pathway_staging"
    ).fetchall()
    for c in clusters:
        gdb.execute(
            "INSERT INTO cluster_narratives (cluster_id, cluster_name, cluster_narrative, generated_at) VALUES (?,?,?,?)",
            (c["cluster_id"], c["cluster_name"], c["cluster_narrative"], c["generated_at"]),
        )
    gdb.commit()
    print(f"Clusters promoted: {len(clusters)}")

    # career_tracks — build old id → new id mapping
    tracks = db.execute(
        "SELECT id, cluster_id, track_name, track_narrative, track_order FROM track_staging ORDER BY id"
    ).fetchall()
    id_map = {}
    for t in tracks:
        cursor = gdb.execute(
            "INSERT INTO career_tracks (cluster_id, track_name, track_narrative, track_order) VALUES (?,?,?,?)",
            (t["cluster_id"], t["track_name"], t["track_narrative"], t["track_order"]),
        )
        id_map[t["id"]] = cursor.lastrowid
    gdb.commit()
    print(f"Tracks promoted: {len(tracks)}")

    # career_track_jobs
    job_refs = db.execute(
        "SELECT track_staging_id, job_id, title, job_order FROM track_jobs_staging ORDER BY track_staging_id, job_order"
    ).fetchall()
    skipped = 0
    inserted = 0
    for j in job_refs:
        new_track_id = id_map.get(j["track_staging_id"])
        if new_track_id is None:
            print(f"  WARNING: track_staging_id {j['track_staging_id']} not in id_map — skipping job {j['job_id']}", file=sys.stderr)
            skipped += 1
            continue
        gdb.execute(
            "INSERT INTO career_track_jobs (career_track_id, job_id, title, job_order) VALUES (?,?,?,?)",
            (new_track_id, j["job_id"], j["title"], j["job_order"]),
        )
        inserted += 1
    gdb.commit()
    print(f"Job references promoted: {inserted}")
    if skipped:
        print(f"Job references skipped (orphaned track ids): {skipped}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    db  = sqlite3.connect(JOBS_DB)
    gdb = sqlite3.connect(GMIOT_DB)
    db.row_factory  = sqlite3.Row
    gdb.row_factory = sqlite3.Row

    apply_staging_fixes(db)
    duplication_check(db)
    create_production_tables(gdb)
    promote(db, gdb)

    # Final verification
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    n_clusters = gdb.execute("SELECT COUNT(*) FROM cluster_narratives").fetchone()[0]
    n_tracks   = gdb.execute("SELECT COUNT(*) FROM career_tracks").fetchone()[0]
    n_jobs     = gdb.execute("SELECT COUNT(*) FROM career_track_jobs").fetchone()[0]
    print(f"cluster_narratives : {n_clusters} rows")
    print(f"career_tracks      : {n_tracks} rows")
    print(f"career_track_jobs  : {n_jobs} rows")

    # Track count per cluster
    print()
    print("Tracks per cluster:")
    for r in gdb.execute("""
        SELECT cn.cluster_id, cn.cluster_name, COUNT(ct.id) as n
        FROM cluster_narratives cn
        LEFT JOIN career_tracks ct ON ct.cluster_id = cn.cluster_id
        GROUP BY cn.cluster_id ORDER BY cn.cluster_id + 0, cn.cluster_id
    """):
        print(f"  {r['cluster_id']:5} {r['cluster_name']:<45} {r['n']} tracks")

    db.close()
    gdb.close()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
