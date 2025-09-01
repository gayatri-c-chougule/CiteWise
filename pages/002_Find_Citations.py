# CiteWise: search embedded chunks across selected ChromaDB collections

import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict

# -----------------------------
# Config
# -----------------------------
load_dotenv()

DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# -----------------------------
# Data structure
# -----------------------------
class Source:
    """
    Container for a retrieved chunk from ChromaDB.
    """
    def __init__(self, filename, id, doc, metadata, distance):
        self.filename = filename
        self.id = id
        self.doc = doc
        self.metadata = metadata or {}
        self.distance = distance

# -----------------------------
# Helper functions
# -----------------------------
def get_sources(results_by_collection: Dict[str, dict]) -> List[Source]:
    """Flatten ChromaDB query results into a list of Source objects."""
    sources: List[Source] = []
    for filename, results in results_by_collection.items():
        # Guard for empty results
        if not results or "ids" not in results or not results["ids"] or not results["ids"][0]:
            continue
        count = len(results["ids"][0])
        for i in range(count):
            sources.append(
                Source(
                    filename=filename,
                    id=results["ids"][0][i],
                    doc=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    distance=results["distances"][0][i],
                )
            )
    return sources

def get_top_k_sources(sources_list: List[Source], num_k: int) -> List[Source]:
    """Sort by distance (ascending) and return top K."""
    if not sources_list:
        return []
    return sorted(sources_list, key=lambda x: x.distance)[:num_k]

# -----------------------------
# ChromaDB helpers
# -----------------------------
def list_all_collections(db_path: str | None = None) -> list[str]:
    """Return all ChromaDB collection names at the given path."""
    path = db_path or DB_PATH
    Path(path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=path)
    return [col.name for col in client.list_collections()]

# -----------------------------
# Embedding/query helpers
# -----------------------------
@st.cache_resource
def get_embedder():
    """Load and cache the sentence-transformer model for query embedding."""
    return SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)

embedder = get_embedder()

def embed_query(query: str):
    """Convert a string query into a numeric embedding vector."""
    q = (query or "").strip()
    if not q:
        return None
    try:
        return embedder.encode(q, convert_to_numpy=True)
    except Exception:
        return None

def retrieve_top_k(query_embedding, k: int, collection_name: str):
    """Query a single ChromaDB collection for top K most similar chunks."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(collection_name)
    if query_embedding is None:
        return {}
    k = max(1, min(int(k), 50))
    return collection.query(query_embeddings=[query_embedding], n_results=k)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧾 CiteWise: Your AI-Powered Research Assistant")
st.caption(f"DB: {Path(DB_PATH).resolve()} • Model: {EMBEDDING_MODEL} ({EMBEDDING_DEVICE})")
st.markdown("---")

# Sidebar: select collections to search
checkbox_items = list_all_collections()
if not checkbox_items:
    st.info("No collections found yet. Go to the upload page to add PDFs.")
    st.stop()

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### Select Sources")
    selected_sources: list[str] = []
    for item in checkbox_items:
        if st.checkbox(item, value=True):  # default: all checked
            selected_sources.append(item)

with col2:
    user_query = st.text_area("Enter your query:", height=100)
    num_k = st.text_input("Number of matches:", value="5")
    find_clicked = st.button("Find Matches")
    output_area = st.empty()

    if find_clicked:
        # --- Minimal validations ---
        if not selected_sources: 
            output_area.warning("Please select at least one source to search.")
            st.stop()

        if not user_query.strip():
            output_area.warning("Please enter a query.")
            st.stop()

        try:
            k = int(num_k)
            if k <= 0:
                raise ValueError
        except ValueError:
            output_area.warning("Please enter a valid positive number for matches.")
            st.stop()

        # Embed query
        embedded_query = embed_query(user_query)
        if embedded_query is None: 
            output_area.error("Could not embed your query. Please try a different query or reload the app.")
            st.stop()

        # Retrieve from selected collections
        results_by_collection: dict = {}
        for name in selected_sources:
            res = retrieve_top_k(embedded_query, k, name)
            if res and res.get("ids") and res["ids"][0]:  
                results_by_collection[name] = res

        # Flatten results and show
        if results_by_collection:
            flat_sources = get_sources(results_by_collection)
            top_hits = get_top_k_sources(flat_sources, k)

            if not top_hits:
                output_area.info("No matches found.")
            else:
                for hit in top_hits:
                    score = f"{hit.distance:.4f}" if isinstance(hit.distance, (int, float)) else str(hit.distance)
                    with st.expander(f"{hit.filename} | Distance: {score}"):
                        pages = hit.metadata.get("page_num", "N/A")
                        st.markdown(f"📎 Page(s): `{pages}`")
                        st.markdown(f"📝 {hit.doc} …")
        else:
            output_area.warning("No results returned from the selected sources.")
