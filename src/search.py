import re
import sys
import time
from pathlib import Path

import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "vector_db"
COLLECTION_NAME = "scalable_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CANDIDATE_COUNT = 100
RESULT_COUNT = 3


def normalize_for_deduplication(text):
    """Treat generated load-test copies as the same underlying passage."""
    text = text.lower()
    text = re.sub(r"load-test record \d+\.\s*", "", text)
    text = re.sub(r"\s*synthetic record identifier:\s*\d+\.?", "", text)
    return " ".join(text.split())


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        print("No chunks found. Run src/ingest.py first.")
        return

    print(f"Search ready! Indexed chunks: {collection.count()}")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            continue

        start = time.perf_counter()

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(CANDIDATE_COUNT, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        unique_results = []
        seen_texts = set()

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            normalized = normalize_for_deduplication(document)

            if normalized in seen_texts:
                continue

            seen_texts.add(normalized)
            unique_results.append((document, metadata, distance))

            if len(unique_results) >= RESULT_COUNT:
                break

        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"\nSearch time (embedding + retrieval): {elapsed_ms:.2f} ms")

        if not unique_results:
            print("No results found.")
        else:
            for rank, (document, metadata, distance) in enumerate(
                unique_results, start=1
            ):
                print(f"\nResult {rank}")
                print(f"Source: {metadata.get('source', 'unknown')}")
                print(f"Cosine distance: {distance:.4f}")
                print(f"Text: {document}")

                if distance > 0.7:
                    print("Note: This may be a weak match; verify relevance.")

        print()


if __name__ == "__main__":
    main()
