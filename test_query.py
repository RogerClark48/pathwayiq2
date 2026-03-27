import sys
import chromadb
import ollama

CHROMA_PATH = r"C:\Dev\pathwayiq\chroma_store"
EXCERPT_LEN = 200

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
jobs_collection = chroma_client.get_collection(name="jobs")


def find_jobs(query, n_results=5):
    print(f"\nQuery: '{query}'")
    print("=" * 60)

    response = ollama.embeddings(model="nomic-embed-text", prompt=query)
    query_embedding = response["embedding"]

    results = jobs_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )

    for i, (metadata, document, distance) in enumerate(zip(
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0],
    )):
        score = round((1 - distance) * 100, 1)  # cosine: distance is 1-similarity
        chunk = metadata.get("chunk", "?")

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

        print(f"\n{i+1}. {metadata['title']}  [{chunk}]")
        print(f"   Score:  {score}%  |  Salary: {salary_str}")
        print(f"   URL:    {metadata['url']}")
        print(f"   Excerpt: {excerpt}")

    print()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "i want to use my art skills, cartoons, drawing"
    find_jobs(query)
