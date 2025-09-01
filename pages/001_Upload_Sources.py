"""CiteWise: Upload PDFs and embed them into a local ChromaDB collection for semantic search."""

# stdlib
import os
import re
from pathlib import Path
import io  

# third-party
import chromadb
import fitz  # PyMuPDF
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv  

# -----------------------
# Load .env
# -----------------------
load_dotenv()  

# -----------------------
# Configuration
# -----------------------
DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db") 
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2") 
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu") 

# Setup text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "],
)

# -----------------------
# Model utils
# -----------------------
@st.cache_resource
def get_embedder():
    """Load and cache the SentenceTransformer embedder."""
    return SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)

embedder = get_embedder()

# -----------------------
# Filesystem utils
# -----------------------
def ensure_db_path():
    Path(DB_PATH).mkdir(parents=True, exist_ok=True)  
    return str(Path(DB_PATH).resolve())

# -----------------------
# PDF handling
# -----------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\(cid:\d+\)", "", text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()

def load_pdf_pages(uploaded_file):
    """Try to open PDF and return pages with cleaned text."""
    try:
        raw = uploaded_file.read()
        if not raw:
            st.error(f"❌ `{uploaded_file.name}` is empty or unreadable.") 
            return []
        doc = fitz.open(stream=raw, filetype="pdf")
    except Exception as e:
        st.error(f"❌ `{uploaded_file.name}` could not be opened as a PDF. Details: {e}")  
        return []

    filename = Path(uploaded_file.name).stem
    pages = []
    for i, page in enumerate(doc):
        raw_text = page.get_text()
        cleaned_text = clean_text(raw_text)
        if cleaned_text:
            pages.append({"page_num": [i + 1], "source": filename, "text": cleaned_text})
    return pages

def split_pages_with_metadata(pages):
    chunks = []
    for i, page in enumerate(pages):
        curr_page_num = page["page_num"][0]
        filename = page["source"]
        text = page["text"]

        if i > 0:
            text = pages[i - 1]["text"][-300:] + " " + text

        split_parts = splitter.split_text(text)
        for j, part in enumerate(split_parts):
            if not part.strip():
                continue
            if i > 0 and j == 0:
                page_nums = [pages[i - 1]["page_num"][0], curr_page_num]
            else:
                page_nums = [curr_page_num]
            chunks.append({
                "page_num": page_nums,
                "source": filename,
                "chunk_id": f"{filename}_p{curr_page_num}_c{j}",
                "text": part
            })
    return chunks

def prep_for_embedding(chunks):
    documents = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {"source": c["source"], "page_num": ",".join(map(str, c["page_num"])), "chunk_id": c["chunk_id"]}
        for c in chunks
    ]
    return ids, documents, metadatas

def embed_file_stream_to_chromadb(uploaded_file):
    ensure_db_path()

    pages = load_pdf_pages(uploaded_file)
    if not pages:
        return 0

    chunks = split_pages_with_metadata(pages)
    if not chunks:
        return 0

    ids, documents, metadatas = prep_for_embedding(chunks)

    try:
        embeddings = embedder.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    except Exception as e:
        st.error(f"❌ Embedding failed for `{uploaded_file.name}`. Details: {e}")  
        return 0

    try:
        filename = Path(uploaded_file.name).stem
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_or_create_collection(name=filename)
        collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    except Exception as e:
        st.error(f"❌ Could not write to ChromaDB for `{uploaded_file.name}`. Details: {e}")  
        return 0

    return len(documents)

def list_all_collections(db_path: str | None = None):
    path = db_path or DB_PATH
    client = chromadb.PersistentClient(path=path)
    return [c.name for c in client.list_collections()]

# -----------------------
# Streamlit UI
# -----------------------
st.title("📚 CiteWise: Your AI-Powered Research Assistant")
st.caption(f"DB path: {Path(DB_PATH).resolve()} • Model: {EMBEDDING_MODEL} ({EMBEDDING_DEVICE})")
st.markdown("---")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
)

if not uploaded_files:  
    st.info("⬆️ Please choose one or more **PDF** files to begin.")
else:
    for f in uploaded_files:
        # re-buffer so PyMuPDF can read
        f_bytes = f.read()
        f_buffer = io.BytesIO(f_bytes)
        class _Wrap:
            def __init__(self, name, b): self.name, self._b = name, b
            def read(self): return self._b.getvalue()
        wrapped = _Wrap(f.name, f_buffer)

        num_chunks = embed_file_stream_to_chromadb(wrapped)
        if num_chunks > 0:
            st.success(f"✅ {f.name}: {num_chunks} chunks embedded into ChromaDB.")

st.markdown("### 📂 Available Embedded Resources")
collections = list_all_collections()
if collections:
    for name in collections:
        st.markdown(f"- `{name}`")
else:
    st.info("No resources found in the database.")
