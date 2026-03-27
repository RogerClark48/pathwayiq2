import chromadb

CHROMA_PATH = r"C:\Dev\pathwayiq\chroma_store"
N_JOBS = 3

client = chromadb.PersistentClient(path=CHROMA_PATH)
courses_collection = client.get_collection(name="courses")
jobs_collection = client.get_collection(name="jobs")


def main():
    # Fetch all overview chunks with their embeddings
    results = courses_collection.get(
        where={"chunk": {"$eq": "overview"}},
        include=["metadatas", "embeddings"],
    )

    # Sort by course name for readable output
    entries = sorted(
        zip(results["metadatas"], results["embeddings"]),
        key=lambda x: x[0]["course_name"],
    )

    total = len(entries)
    print(f"Cross-matching {total} courses against jobs collection\n")
    print("=" * 70)

    for i, (meta, vector) in enumerate(entries, 1):
        print(f"\n[{i}/{total}]  {meta['course_name']}")
        print(f"  Provider: {meta['provider']}  |  Level: {meta['level']}  |  Qual: {meta['qualification_type']}")
        print(f"  SSA: {meta['ssa_category']}")
        print()

        job_results = jobs_collection.query(
            query_embeddings=[vector],
            n_results=N_JOBS,
            include=["metadatas", "distances"],
        )

        for j, (jmeta, distance) in enumerate(zip(
            job_results["metadatas"][0],
            job_results["distances"][0],
        ), 1):
            score = round((1 - distance) * 100, 1)
            sal_min = jmeta.get("salary_min", 0.0)
            sal_max = jmeta.get("salary_max", 0.0)
            currency = jmeta.get("salary_currency", "")
            if sal_min or sal_max:
                salary_str = f"{currency}{int(sal_min):,}–{currency}{int(sal_max):,}"
            else:
                salary_str = "salary not listed"

            print(f"  {j}. {jmeta['title']}")
            print(f"     Score: {score}%  |  {salary_str}")

        print()
        print("-" * 70)

    print(f"\nDone. {total} courses processed.")


if __name__ == "__main__":
    main()
