from src.local_llm import load_local_llm, generate_answer
import sys
import time
from pathlib import Path

import pysqlite3

# Use the SQLite version required by this project's ChromaDB setup.
sys.modules["sqlite3"] = pysqlite3

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
DB_DIR = PROJECT_ROOT / "vector_db"
COLLECTION_NAME = "scalable_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(
    page_title="Scalable Vector Search",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource
def load_search_resources():
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    return model, collection


@st.cache_resource
def load_answer_model():
    return load_local_llm()



st.title("🔎 Scalable Vector Search")
st.caption(
    "Search your indexed technical documents using local semantic embeddings."
)

with st.sidebar:
    st.header("Index information")
    st.write(f"**Database:** `{DB_DIR}`")
    st.write(f"**Embedding model:** `{MODEL_NAME}`")

    if st.button("Refresh index count"):
        st.rerun()

try:
    model, collection = load_search_resources()
except Exception as error:
    st.error(f"Could not load the search index: {error}")
    st.info(
        "Check that vector_db/ contains the scalable_documents collection. "
        "If needed, run your project's ingestion script first."
    )
    st.stop()

st.metric("Indexed chunks", f"{collection.count():,}")
st.divider()

st.subheader("Ask a question")

query = st.text_input(
    "Search query",
    placeholder="Example: How do Kubernetes readiness and liveness probes differ?",
)

result_count = st.slider(
    "Number of results",
    min_value=1,
    max_value=10,
    value=3,
)

search_clicked = st.button(
    "Search documents",
    type="primary",
    disabled=not query.strip(),
)

if search_clicked:
    started = time.perf_counter()

    with st.spinner("Searching your documents..."):
        query_embedding = model.encode(
            [query.strip()],
            convert_to_numpy=True,
        ).tolist()

        total = collection.count()

        if total == 0:
            st.warning("Your index is empty. Ingest some documents first.")
        else:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(100, total),
                include=["documents", "metadatas", "distances"],
            )

            unique_results = []
            seen_texts = set()

            for document, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Hide repeated load-test copies in the displayed results.
                import re

                normalized = document.lower()
                normalized = re.sub(
                    r"load-test record \d+\.\s*", "", normalized
                )
                normalized = re.sub(
                    r"\s*synthetic record identifier:\s*\d+\.?",
                    "",
                    normalized,
                )
                normalized = " ".join(normalized.split())

                if normalized in seen_texts:
                    continue

                seen_texts.add(normalized)
                unique_results.append((document, metadata, distance))

                if len(unique_results) >= result_count:
                    break

            elapsed_ms = (time.perf_counter() - started) * 1000

            st.success(
                f"Search completed in {elapsed_ms:.2f} ms "
                "(query embedding + retrieval)."
            )

            if not unique_results:
                st.warning("No results found.")
            else:
                st.subheader("🤖 AI-generated answer")

                passages = [
                    document
                    for document, metadata, distance in unique_results
                ]

                try:
                    with st.spinner("Generating answer locally..."):
                        tokenizer, llm = load_answer_model()
                        answer = generate_answer(
                            query.strip(),
                            passages,
                            tokenizer,
                            llm,
                        )
                    st.info(answer)
                except Exception as error:
                    st.error(f"AI generation failed: {error}")

                st.caption(
                    "Verify the generated answer against the source passages below."
                )
                st.subheader("📚 Supporting documents")

                for rank, (document, metadata, distance) in enumerate(
                    unique_results, start=1
                ):
                    with st.container(border=True):
                        st.markdown(f"**Result {rank}**")
                        st.caption(
                            f"Source: {metadata.get('source', 'unknown')} "
                            f"· Cosine distance: {distance:.4f}"
                        )
                        st.write(document)
