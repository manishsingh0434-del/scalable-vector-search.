
import argparse
import re
import sys
import time
from pathlib import Path

import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "vector_db"

COLLECTION_NAME = "scalable_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32

def split_into_chunks(text, max_words=100):
    paragraphs = [
        " ".join(part.split())
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]

    # Attach a topic label to the following technical paragraph.
    merged = []
    pending_topic = None

    for paragraph in paragraphs:
        if paragraph.startswith("Topic:"):
            pending_topic = paragraph
            continue

        if pending_topic:
            paragraph = f"{pending_topic}. {paragraph}"
            pending_topic = None

        merged.append(paragraph)

    if pending_topic:
        merged.append(pending_topic)

    chunks = []

    for paragraph in merged:
        words = paragraph.split()

        for start in range(0, len(words), max_words):
            chunk = " ".join(words[start:start + max_words])
            if chunk:
                chunks.append(chunk)

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally index text files into ChromaDB."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of text files to index (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of chunks embedded per batch (default: 32).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate only this project's collection.",
    )
    args = parser.parse_args()

    if args.limit < 1 or args.batch_size < 1:
        parser.error("--limit and --batch-size must be positive.")

    files = sorted(DATA_DIR.glob("*.txt"))[:args.limit]

    if not files:
        print(f"No .txt files found in {DATA_DIR}")
        return

    print("Loading local embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_DIR))

    if args.reset:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print("Removed the existing project collection.")
        except Exception as error:
            # Chroma versions differ in their missing-collection errors.
            print(f"No existing collection to reset (or reset issue): {error}")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    started = time.perf_counter()
    total_chunks = 0

    for file_number, file_path in enumerate(files, start=1):
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            print(f"Skipping {file_path.name}: {error}")
            continue

        chunks = split_into_chunks(text)

        # Remove old entries for this source before inserting its current chunks.
        collection.delete(where={"source": file_path.name})

        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", file_path.stem)

        for offset in range(0, len(chunks), args.batch_size):
            batch = chunks[offset:offset + args.batch_size]

            embeddings = model.encode(
                batch,
                batch_size=args.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()

            ids = [
                f"{safe_name}_chunk_{offset + i}"
                for i in range(len(batch))
            ]

            metadatas = [
                {
                    "source": file_path.name,
                    "chunk_index": offset + i,
                }
                for i in range(len(batch))
            ]

            collection.upsert(
                ids=ids,
                documents=batch,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            total_chunks += len(batch)

        if file_number % 10 == 0 or file_number == len(files):
            elapsed = time.perf_counter() - started
            print(
                f"Files: {file_number}/{len(files)} | "
                f"Chunks this run: {total_chunks} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    elapsed = time.perf_counter() - started

    print("\nIngestion benchmark")
    print(f"Files selected: {len(files)}")
    print(f"Chunks indexed this run: {total_chunks}")
    print(f"Total chunks in collection: {collection.count()}")
    print(f"Elapsed indexing time: {elapsed:.2f} seconds")
    print(f"Chunk throughput: {total_chunks / elapsed:.2f} chunks/sec")
    print(f"Database directory: {DB_DIR}")


if __name__ == "__main__":
    main()
