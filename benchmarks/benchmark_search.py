
import sys
import time
import statistics
from pathlib import Path

import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "vector_db"
COLLECTION_NAME = "scalable_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

QUERIES = [
    "How can Linux tools diagnose system problems?",
    "How do vector embeddings support semantic retrieval?",
    "What metrics help monitor application performance?",
]

WARMUP_RUNS = 2
MEASURED_RUNS = 10


def summarize(label, values):
    ordered = sorted(values)
    p95_index = min(
        len(ordered) - 1,
        int(0.95 * len(ordered)),
    )

    print(f"{label}:")
    print(f"  Mean:   {statistics.mean(values):.2f} ms")
    print(f"  Median: {statistics.median(values):.2f} ms")
    print(f"  P95:    {ordered[p95_index]:.2f} ms")


def main():
    print("Loading local embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Indexed chunks: {collection.count()}")

    for query in QUERIES:
        print(f"\nQuery: {query}")

        # Warm up model and database before recording measurements.
        for _ in range(WARMUP_RUNS):
            embedding = model.encode(
                [query],
                convert_to_numpy=True,
            ).tolist()

            collection.query(
                query_embeddings=embedding,
                n_results=min(3, collection.count()),
                include=["documents", "distances"],
            )

        embedding_times = []
        retrieval_times = []
        top_result = None

        for _ in range(MEASURED_RUNS):
            start = time.perf_counter()
            embedding = model.encode(
                [query],
                convert_to_numpy=True,
            ).tolist()
            embedding_times.append(
                (time.perf_counter() - start) * 1000
            )

            start = time.perf_counter()
            results = collection.query(
                query_embeddings=embedding,
                n_results=min(3, collection.count()),
                include=["documents", "distances"],
            )
            retrieval_times.append(
                (time.perf_counter() - start) * 1000
            )

            top_result = results["documents"][0][0]

        summarize("Embedding latency", embedding_times)
        summarize("Retrieval latency", retrieval_times)
        print(f"Top result: {top_result}")


if __name__ == "__main__":
    main()
