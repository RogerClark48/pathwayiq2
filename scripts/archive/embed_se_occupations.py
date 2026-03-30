"""
embed_se_occupations.py — Embed SE occupation records into se_data.db.
Text: name + typical_job_titles (pipe-separated). Name only if titles are NULL.
Stored as float32 BLOB in se_occupations.embedding.
Safe to re-run — skips rows where embedding IS NOT NULL.
"""

import os
import sys
import sqlite3
import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()

ROOT    = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT, "se_data.db")
BATCH   = 50
MODEL   = "voyage-3.5"
DIMS    = 1024

vo = voyageai.Client()


def ensure_column(conn):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(se_occupations)").fetchall()]
    if "embedding" not in cols:
        conn.execute("ALTER TABLE se_occupations ADD COLUMN embedding BLOB")
        conn.commit()
        print("Added embedding column.")


def build_text(name, typical_job_titles):
    if typical_job_titles:
        titles = " | ".join(t.strip() for t in typical_job_titles.split("|"))
        return f"{name} | {titles}"
    return name


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    ensure_column(conn)

    rows = conn.execute(
        "SELECT std_code, name, typical_job_titles FROM se_occupations WHERE embedding IS NULL"
    ).fetchall()
    total = len(rows)
    print(f"{total} occupations to embed (skipping already-embedded rows).\n")

    n_embedded = 0
    n_failed   = 0

    for batch_start in range(0, total, BATCH):
        batch = rows[batch_start:batch_start + BATCH]
        std_codes = [r[0] for r in batch]
        texts     = [build_text(r[1], r[2]) for r in batch]

        try:
            result = vo.embed(texts, model=MODEL, input_type="document", output_dimension=DIMS)
            vectors = result.embeddings

            for std_code, vector in zip(std_codes, vectors):
                blob = np.array(vector, dtype=np.float32).tobytes()
                conn.execute(
                    "UPDATE se_occupations SET embedding = ? WHERE std_code = ?",
                    (blob, std_code)
                )
            conn.commit()
            n_embedded += len(batch)

        except Exception as e:
            print(f"  ERROR batch {batch_start}–{batch_start + len(batch)}: {e}", flush=True)
            n_failed += len(batch)

        done = batch_start + len(batch)
        print(f"[{done}/{total}] {n_embedded} embedded, {n_failed} failed", flush=True)

    print(f"\n=== Complete ===")
    print(f"Embedded : {n_embedded}")
    print(f"Failed   : {n_failed}")

    # Verification
    row = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) FROM se_occupations"
    ).fetchone()
    print(f"\nVerification: {row[1]}/{row[0]} rows have embeddings")

    sample = conn.execute(
        "SELECT std_code, name, embedding FROM se_occupations WHERE embedding IS NOT NULL LIMIT 1"
    ).fetchone()
    if sample:
        vec = np.frombuffer(sample[2], dtype=np.float32)
        print(f"Sample: {sample[0]} '{sample[1]}' — shape={vec.shape}, dtype={vec.dtype}")

    conn.close()
    print("Done.")
