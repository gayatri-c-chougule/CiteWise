from __future__ import annotations

import chromadb
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

# -----------------------------
# Data structure
# -----------------------------
class Source:
    """
    Container for a retrieved chunk from ChromaDB.
    """
    def __init__(self, filename: str, id: str, doc: str, metadata: dict, distance: float):
        self.filename = filename
        self.id = id
        self.doc = doc
        self.metadata = metadata or {}
        self.distance = distance

# -----------------------------
# Core helpers (pure Python)
# -----------------------------
def build_embedder(model_name: str, device: str) -> SentenceTransformer:
    """Create an embedding model (no Streamlit cache here)."""
    return SentenceTransformer(model_name, device=device)

def list_all_collections(db_path: str) -> list[str]:
    """Return all ChromaDB collection names at the given path."""
    Path(db_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    return [col.name for col in client.list_collections()]

def embed_query(embedder: SentenceTransformer, query: str):
    """Convert a string query into a numeric embedding vector (or None)."""
    q = (query or "").strip()
    if not q:
        return None
    try:
        return embedder.encode(q, convert_to_numpy=True)
    except Exception:
        return None

def retrieve_top_k(db_path: str, query_embedding, k: int, collection_name: str) -> Dict[str, Any]:
    """Query a single ChromaDB collection for top K most similar chunks."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)
    if query_embedding is None:
        return {}
    k = max(1, min(int(k), 50))
    return collection.query(query_embeddings=[query_embedding], n_results=k)

def get_sources(results_by_collection: Dict[str, dict]) -> List[Source]:
    """Flatten ChromaDB query results into a list of Source objects."""
    sources: List[Source] = []
    for filename, results in results_by_collection.items():
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

def run_search_logic(
    embedder: SentenceTransformer,
    db_path: str,
    user_query: str,
    num_k: str,
    selected_sources: list[str],
) -> dict:
    """
    Core search logic for CiteWise.
    Returns a dict with either 'error', 'warning', 'info', or 'results'.
    """
    # --- Validations ---
    if not selected_sources:
        return {"warning": "Please select at least one source to search."}

    if not user_query.strip():
        return {"warning": "Please enter a query."}

    try:
        k = int(num_k)
        if k <= 0:
            raise ValueError
    except ValueError:
        return {"warning": "Please enter a valid positive number for matches."}

    # --- Embed query ---
    qvec = embed_query(embedder, user_query)
    if qvec is None:
        return {"error": "Could not embed your query. Please try a different query or reload the app."}

    # --- Retrieve from selected collections ---
    results_by_collection: dict = {}
    for name in selected_sources:
        res = retrieve_top_k(db_path, qvec, k, name)
        if res and res.get("ids") and res["ids"][0]:
            results_by_collection[name] = res

    # --- Flatten results ---
    if not results_by_collection:
        return {"warning": "No results returned from the selected sources."}

    flat_sources = get_sources(results_by_collection)
    top_hits = get_top_k_sources(flat_sources, k)

    if not top_hits:
        return {"info": "No matches found."}

    # --- Success ---
    return {"results": top_hits}
