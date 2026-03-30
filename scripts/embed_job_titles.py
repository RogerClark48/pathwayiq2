"""
embed_job_titles.py — Embed job titles from job_roles_asset.db.
Text: title only (title-to-title matching against SE occupations).
Stored as float32 BLOB in jobs.title_embedding.
Safe to re-run — skips rows where title_embedding IS NOT NULL.
"""

import os
import sys
import sqlite3
import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()

ROOT    = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT, "job_roles_asset.db")
BATCH   = 50
MODEL   = "voyage-3.5"
DIMS    = 1024

vo = voyageai.Client()


def ensure_column(conn):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
    if "title_embedding" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN title_embedding BLOB")
        conn.commit()
        print("Added title_embedding column.")


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    ensure_column(conn)

    rows = conn.execute(
        "SELECT id, title FROM jobs WHERE title_embedding IS NULL"
    ).fetchall()
    total = len(rows)
    print(f"{total} jobs to embed (skipping already-embedded rows).\n")

    n_embedded = 0
    n_null     = 0
    n_failed   = 0

    for batch_start in range(0, total, BATCH):
        batch = rows[batch_start:batch_start + BATCH]

        # Split out null-title records — nothing to embed
        to_embed   = [(r[0], r[1]) for r in batch if r[1]]
        null_ids   = [r[0] for r in batch if not r[1]]
        n_null    += len(null_ids)

        if to_embed:
            ids   = [r[0] for r in to_embed]
            texts = [r[1] for r in to_embed]

            try:
                result = vo.embed(texts, model=MODEL, input_type="document", output_dimension=DIMS)
                vectors = result.embeddings

                for job_id, vector in zip(ids, vectors):
                    blob = np.array(vector, dtype=np.float32).tobytes()
                    conn.execute(
                        "UPDATE jobs SET title_embedding = ? WHERE id = ?",
                        (blob, job_id)
                    )
                conn.commit()
                n_embedded += len(to_embed)

            except Exception as e:
                print(f"  ERROR batch {batch_start}–{batch_start + len(batch)}: {e}", flush=True)
                n_failed += len(to_embed)

        done = batch_start + len(batch)
        print(f"[{done}/{total}] {n_embedded} embedded, {n_null} null title, {n_failed} failed", flush=True)

    print(f"\n=== Complete ===")
    print(f"Embedded   : {n_embedded}")
    print(f"Null title : {n_null}")
    print(f"Failed     : {n_failed}")

    # Verification
    row = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN title_embedding IS NOT NULL THEN 1 ELSE 0 END) FROM jobs"
    ).fetchone()
    print(f"\nVerification: {row[1]}/{row[0]} rows have title embeddings")

    sample = conn.execute(
        "SELECT id, title, title_embedding FROM jobs WHERE title_embedding IS NOT NULL LIMIT 1"
    ).fetchone()
    if sample:
        vec = np.frombuffer(sample[2], dtype=np.float32)
        print(f"Sample: id={sample[0]} '{sample[1]}' — shape={vec.shape}, dtype={vec.dtype}")

    conn.close()
    print("Done.")
