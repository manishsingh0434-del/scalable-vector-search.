import sys
from pathlib import Path

import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "vector_db"
COLLECTION_NAME = "scalable_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CASES = [
    ("Which Linux commands help diagnose high CPU and memory usage?", "Linux", ["top", "ps", "cpu", "memory"]),
    ("How can I inspect disk space and filesystem capacity?", "Linux", ["df", "disk", "filesystem"]),
    ("How do embeddings and cosine distance enable semantic search?", "Vector Search", ["embedding", "cosine", "similarity"]),
    ("How should retrieval quality be measured using Recall at k?", "Vector Search", ["recall", "retrieval", "relevant"]),
    ("How do I measure p95 latency, error rates, and application health?", "Observability", ["p95", "latency", "error"]),
    ("How can logs metrics and traces help investigate an outage?", "Observability", ["logs", "metrics", "traces"]),
    ("How do Docker images and multi-stage builds reduce image size?", "Docker", ["images", "multi-stage", "build"]),
    ("How do Kubernetes readiness probes and liveness probes differ?", "Kubernetes", ["readiness", "liveness", "traffic"]),
    ("How can batch size and checkpoints make large ingestion jobs safer?", "Batch Processing", ["batch", "checkpoint", "ingestion"]),
    ("How do database indexes and query plans improve query performance?", "Databases", ["index", "query", "performance"]),
    ("How do Python generators process large datasets with less memory?", "Python", ["generator", "memory", "yield"]),
    ("Why use virtual environments and pinned dependencies in Python?", "Python", ["virtual environment", "dependencies", "versions"]),
]

TOPICS = [
    "Python", "Docker", "Kubernetes", "Vector Search",
    "Batch Processing", "Observability", "Linux", "Databases",
]


def topic_from_source(source):
    if not source or not source.startswith("synthetic_") or not source.endswith(".txt"):
        return None
    try:
        index = int(source[len("synthetic_"):-len(".txt")])
        return TOPICS[index % len(TOPICS)]
    except ValueError:
        return None


def main():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        print("No indexed chunks. Run python src/ingest.py first.")
        return

    print("Loading local embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    topic_hits_1 = 0
    topic_hits_3 = 0
    concept_hits_3 = 0
    reciprocal_ranks = []

    print(f"\nIndexed chunks: {collection.count()}")
    print(f"Evaluation queries: {len(CASES)}\n")

    for query, expected_topic, concepts in CASES:
        embedding = model.encode([query], convert_to_numpy=True).tolist()
        result = collection.query(
            query_embeddings=embedding,
            n_results=min(3, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]

        ranks = [
            rank for rank, metadata in enumerate(metadatas, 1)
            if topic_from_source(metadata.get("source")) == expected_topic
        ]
        concept_match = any(
            any(term in document.lower() for term in concepts)
            for document in documents
        )

        hit1 = bool(ranks and ranks[0] == 1)
        hit3 = bool(ranks)
        topic_hits_1 += int(hit1)
        topic_hits_3 += int(hit3)
        concept_hits_3 += int(concept_match)
        reciprocal_ranks.append(1 / ranks[0] if ranks else 0.0)

        source = metadatas[0].get("source", "unknown") if metadatas else "none"
        text = documents[0][:150] if documents else "(no result)"

        print(f"Query: {query}")
        print(f"Expected topic: {expected_topic}")
        print(f"Top source: {source}")
        print(f"Topic match at rank 1: {'YES' if hit1 else 'NO'}")
        print(f"Topic match in top 3: {'YES' if hit3 else 'NO'}")
        print(f"Concept match in top 3: {'YES' if concept_match else 'NO'}")
        print(f"Top result: {text}\n")

    total = len(CASES)
    print("EVALUATION SUMMARY")
    print("-" * 35)
    print(f"Topic Recall@1:      {topic_hits_1 / total:.1%} ({topic_hits_1}/{total})")
    print(f"Topic Recall@3:      {topic_hits_3 / total:.1%} ({topic_hits_3}/{total})")
    print(f"Concept hit rate@3:  {concept_hits_3 / total:.1%} ({concept_hits_3}/{total})")
    print(f"Topic-based MRR:     {sum(reciprocal_ranks) / total:.3f}")
    print("\nThese are proxy metrics, not human-judged relevance scores.")


if __name__ == "__main__":
    main()
