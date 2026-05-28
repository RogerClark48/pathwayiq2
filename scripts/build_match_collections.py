"""
Build Chroma collections for v3 match_chunks.
Creates two new collections:
  - match_courses: course match_chunks from futurefinder.sqlite
  - match_jobs:    job match_chunks from job_roles_asset.db

Usage:
  python build_match_collections.py             # skip if already populated
  python build_match_collections.py --recreate  # delete and rebuild both
"""

import sqlite3
import os
import argparse
import chromadb
import voyageai
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COURSES_DB  = os.path.join(PROJECT_ROOT, "futurefinder.sqlite")
JOBS_DB     = os.path.join(PROJECT_ROOT, "job_roles_asset.db")
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_store")

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
vo = voyageai.Client()

COLLECTION_COURSES = "match_courses"
COLLECTION_JOBS    = "match_jobs"

EMBED_BATCH  = 128
CHROMA_BATCH = 500


def batch_embed(texts, label=""):
    vectors = []
    total = len(texts)
    for i in range(0, total, EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        result = vo.embed(batch, model="voyage-3.5",
                          input_type="document", output_dimension=1024)
        vectors.extend(result.embeddings)
        print(f"  {label}Embedded {min(i + EMBED_BATCH, total)}/{total}", end="\r")
    print()
    return vectors


def chroma_add(collection, ids, documents, embeddings, metadatas):
    for i in range(0, len(ids), CHROMA_BATCH):
        collection.add(
            ids=ids[i:i + CHROMA_BATCH],
            documents=documents[i:i + CHROMA_BATCH],
            embeddings=embeddings[i:i + CHROMA_BATCH],
            metadatas=metadatas[i:i + CHROMA_BATCH],
        )


def build_courses(client, recreate):
    if recreate:
        try:
            client.delete_collection(COLLECTION_COURSES)
            print(f"Deleted '{COLLECTION_COURSES}'.")
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=COLLECTION_COURSES,
        metadata={"hnsw:space": "cosine"},
    )

    if col.count() > 0 and not recreate:
        print(f"'{COLLECTION_COURSES}' already has {col.count()} docs — skipping "
              f"(use --recreate to rebuild).")
        return

    conn = sqlite3.connect(COURSES_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT c.course_id, c.course_title, p.provider_name,
               c.level, c.ssa_code, c.qual_type, c.mode, c.campus,
               c.match_chunk
        FROM courses c
        JOIN providers p ON c.provider_id = p.provider_id
        WHERE c.is_active = 1 AND c.match_chunk IS NOT NULL
        ORDER BY c.course_id
    """).fetchall()
    conn.close()

    print(f"\nBuilding '{COLLECTION_COURSES}': {len(rows)} courses ...")
    texts    = [r["match_chunk"] for r in rows]
    vectors  = batch_embed(texts, "courses ")
    ids      = [str(r["course_id"]) for r in rows]
    metas    = [
        {
            "title":     r["course_title"],
            "provider":  r["provider_name"],
            "level":     r["level"] or 0,
            "ssa_code":  r["ssa_code"] or 0,
            "qual_type": r["qual_type"] or "",
            "mode":      r["mode"] or "",
            "campus":    r["campus"] or "",
        }
        for r in rows
    ]
    chroma_add(col, ids, texts, vectors, metas)
    print(f"  Stored {len(ids)} course vectors in '{COLLECTION_COURSES}'.")


def build_jobs(client, recreate):
    if recreate:
        try:
            client.delete_collection(COLLECTION_JOBS)
            print(f"Deleted '{COLLECTION_JOBS}'.")
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=COLLECTION_JOBS,
        metadata={"hnsw:space": "cosine"},
    )

    if col.count() > 0 and not recreate:
        print(f"'{COLLECTION_JOBS}' already has {col.count()} docs — skipping "
              f"(use --recreate to rebuild).")
        return

    conn = sqlite3.connect(JOBS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, title, source, level, match_chunk
        FROM jobs
        WHERE is_active = 1 AND match_chunk IS NOT NULL
        ORDER BY id
    """).fetchall()
    conn.close()

    print(f"\nBuilding '{COLLECTION_JOBS}': {len(rows)} jobs ...")
    texts   = [r["match_chunk"] for r in rows]
    vectors = batch_embed(texts, "jobs ")
    ids     = [str(r["id"]) for r in rows]
    metas   = [
        {
            "title":  r["title"],
            "source": r["source"],
            "level":  r["level"] or 0,
        }
        for r in rows
    ]
    chroma_add(col, ids, texts, vectors, metas)
    print(f"  Stored {len(ids)} job vectors in '{COLLECTION_JOBS}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and rebuild both collections from scratch")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    build_courses(client, args.recreate)
    build_jobs(client, args.recreate)
    print("\nDone.")


if __name__ == "__main__":
    main()
