# Scalable Vector Search

A local semantic search system that indexes technical documents using
sentence embeddings and retrieves relevant passages using ChromaDB.

## Features

- Local semantic embeddings using `all-MiniLM-L6-v2`
- Persistent vector storage with ChromaDB
- Configurable document chunking and ingestion batches
- Incremental indexing and project-scoped collection reset
- Duplicate-aware search result filtering
- Search latency and retrieval evaluation scripts
- Automated unit tests for document chunking
- Synthetic dataset generator for load testing

## Architecture

```text
Text files
    |
    v
Document ingestion and chunking
    |
    v
Sentence Transformers embeddings
    |
    v
ChromaDB persistent vector index
    |
    v
Query embedding -> Vector retrieval
    |
    v
Deduplicated search results