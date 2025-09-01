"""CiteWise — Home page for the Streamlit app."""

import streamlit as st

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="CiteWise",  # browser tab title
    page_icon="📚",          # favicon / emoji
    layout="wide"            # wide layout for better readability
)

# -----------------------------
# Home page content (Markdown)
# -----------------------------
HOME_INTRO_MD = """
### 🔍 What is CiteWise?

**CiteWise** is a local AI-powered research companion. It lets you:
- Upload PDF research papers  
- Extract, clean, and split text into searchable chunks  
- Generate embeddings using **all-mpnet-base-v2**  
- Store everything locally in a **ChromaDB** vector database  
- Search across your documents using semantic similarity  
- See results with **source filename and page numbers**  

---

### 🚀 How to Use

1. **📄 Upload PDFs**  
   - Add one or more PDF files  
   - Text is extracted per page, cleaned, chunked  
   - Embeddings are generated and stored locally in ChromaDB  

2. **🔍 Search Documents**  
   - Select which uploaded documents to include  
   - Enter your question in plain English  
   - Get top relevant matches with **page numbers and text snippets**  

3. **📂 View Sources**  
   - Browse a list of all embedded document collections in your local store  

---

### 💡 Why Use CiteWise?

- **Local-first & privacy-friendly**: Everything stays on your machine  
- **Accurate semantic search** with high-quality embeddings  
- **Simple local setup**: `pip install -r requirements.txt` → `streamlit run ...`
"""

# -----------------------------
# Render home page
# -----------------------------
def render_home() -> None:
    """
    Display the home page content for the CiteWise app.

    This includes:
        - Title
        - Introduction & explanation of features
        - Quick usage guide
        - Benefits of using CiteWise
    """
    st.title("Welcome to CiteWise")
    st.markdown(HOME_INTRO_MD)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    render_home()
