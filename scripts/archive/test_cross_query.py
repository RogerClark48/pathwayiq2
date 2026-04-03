import sys
import chromadb

CHROMA_PATH = r"C:\Dev\pathwayiq\chroma_store"
EXCERPT_LEN = 200

client = chromadb.PersistentClient(path=CHROMA_PATH)
courses_collection = client.get_collection(name="courses")
jobs_collection = client.get_collection(name="jobs")


def list_courses():
    results = courses_collection.get(
        where={"chunk": {"$eq": "overview"}},
        include=["metadatas"],
    )
    names = sorted(m["course_name"] for m in results["metadatas"])
    print(f"\n{len(names)} courses in collection:\n")
    for i, name in enumerate(names, 1):
        print(f"  {i:3d}.  {name}")
    print()
    return names


def resolve_course_name(arg, names):
    """Accept a list number or case-insensitive partial name match."""
    # Numeric index
    if arg.isdigit():
        idx = int(arg) - 1
        if 0 <= idx < len(names):
            return names[idx]
        print(f"Number {arg} out of range (1–{len(names)})")
        return None
    # Partial name match
    arg_lower = arg.lower()
    matches = [n for n in names if arg_lower in n.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Ambiguous — {len(matches)} courses match '{arg}':")
        for m in matches:
            print(f"  {m}")
        return None
    print(f"No course found matching '{arg}'")
    return None


def cross_query(course_name, n_results=5):
    # Find the overview chunk for this course and lift its embedding directly
    results = courses_collection.get(
        where={"$and": [
            {"course_name": {"$eq": course_name}},
            {"chunk": {"$eq": "overview"}},
        ]},
        include=["embeddings", "metadatas"],
    )

    if not results["ids"]:
        print(f"No overview chunk found for course: '{course_name}'")
        print("Run list mode (no args) to see available course names.")
        return

    meta = results["metadatas"][0]
    vector = results["embeddings"][0]

    print(f"\nCourse:   {meta['course_name']}")
    print(f"Provider: {meta['provider']}")
    print(f"Level:    {meta['level']}  |  Qual: {meta['qualification_type']}  |  SSA: {meta['ssa_category']}")
    print(f"\nTop {n_results} matching jobs:")
    print("=" * 60)

    job_results = jobs_collection.query(
        query_embeddings=[vector],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )

    for i, (metadata, document, distance) in enumerate(zip(
        job_results["metadatas"][0],
        job_results["documents"][0],
        job_results["distances"][0],
    )):
        score = round((1 - distance) * 100, 1)

        sal_min = metadata.get("salary_min", 0.0)
        sal_max = metadata.get("salary_max", 0.0)
        currency = metadata.get("salary_currency", "")
        if sal_min or sal_max:
            salary_str = f"{currency}{int(sal_min):,} – {currency}{int(sal_max):,}"
        else:
            salary_str = "not listed"

        excerpt = document.replace("\n", " ").strip()
        if len(excerpt) > EXCERPT_LEN:
            excerpt = excerpt[:EXCERPT_LEN].rsplit(" ", 1)[0] + "…"

        print(f"\n{i+1}. {metadata['title']}  [{metadata.get('chunk', '?')}]")
        print(f"   Score:   {score}%  |  Salary: {salary_str}")
        print(f"   URL:     {metadata['url']}")
        print(f"   Excerpt: {excerpt}")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        list_courses()
    else:
        names = list_courses()
        course_name = resolve_course_name(" ".join(sys.argv[1:]), names)
        if course_name:
            cross_query(course_name)
