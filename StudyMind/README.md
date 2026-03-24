# 📖 StudyMind — AI-Powered Study Assistant

> Upload your study materials. Ask questions. Get instant, context-grounded answers 

##  Project Overview

**StudyMind** is a Retrieval-Augmented Generation (RAG) application that transforms your study documents (PDFs and text files) into an interactive AI knowledge base. Upload your notes, textbooks, or research papers, then ask natural-language questions or request topic summaries — StudyMind retrieves the most relevant passages from your documents and uses an LLM to generate grounded, accurate answers.

### Problem Statement

Students and researchers often struggle to quickly extract relevant information from large volumes of study material. Searching through PDFs manually is slow and ineffective. Traditional keyword search misses semantically related content. StudyMind solves this by combining **semantic vector search** (via Endee) with **LLM-powered synthesis** (via Groq) to give you intelligent, context-aware answers drawn directly from your own documents.

---

##  Features

- ** Document Upload** — Upload `.pdf` and `.txt` study materials via drag-and-drop or file picker
- ** Semantic Search** — Vector similarity search using Endee finds the most relevant passages even without exact keyword matches
- ** RAG Q&A** — Ask any question; LLaMA 3 answers using only your uploaded content as context, with source attribution
- ** Smart Summarization** — Generate concise summaries of any topic present in your documents
- ** Document Management** — View and remove uploaded documents from the knowledge base
- ** Clean UI** — Dark-themed, responsive single-page frontend with real-time status indicator

---

##  System Design

```
┌──────────────────────────────────────────────────────────────────┐
│                          Browser (UI)                            │
│         Upload PDF/TXT  │  Ask Question  │  Get Summary          │
└────────────┬────────────┴───────┬────────┴────────────────────────┘
             │                   │
             ▼                   ▼
┌────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                              │
│  POST /upload   GET /ask   GET /summary   GET /documents        │
│                                                                │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────────┐  │
│  │  ingest.py   │    │  retriever.py │    │ study_chain.py  │  │
│  │              │    │               │    │                 │  │
│  │ Extract text │    │ Embed query   │    │ Build prompt    │  │
│  │ Chunk text   │    │ Vector search │───▶│ Call Groq LLM  │  │
│  │ Embed chunks │    │ Return top-k  │    │ Return answer   │  │
│  │ Upsert Endee │    │               │    │                 │  │
│  └──────┬───────┘    └───────┬───────┘    └─────────────────┘  │
└─────────┼──────────────────┼─────────────────────────────────-─┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Endee Vector Database                         │
│          Index: studymind_index  │  Dimension: 384              │
│          Space: cosine           │  Precision: INT8             │
│          AVX2-accelerated HNSW vector search                    │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│             Groq API  (LLaMA 3.1 8B Instant)                    │
│     Receives: retrieved context + user question                 │
│     Returns:  grounded natural-language answer                  │
└─────────────────────────────────────────────────────────────────┘
```

### Document Ingestion Pipeline

1. **Text Extraction** — PDFs are parsed using `pypdf`; plain text files are decoded directly
2. **Chunking** — Text is split into overlapping 400-character chunks (80-character overlap) to preserve context across chunk boundaries
3. **Embedding** — Each chunk is embedded using `all-MiniLM-L6-v2` (384-dimensional dense vectors) via `sentence-transformers`
4. **Storage** — Vectors and metadata (text, source filename, chunk index) are batch-upserted into Endee in groups of 100

### Query Pipeline

1. **Query Embedding** — The user's question is embedded with the same `all-MiniLM-L6-v2` model
2. **Semantic Retrieval** — Endee performs cosine-similarity search and returns the top-k most relevant chunks
3. **Prompt Construction** — Retrieved chunks are formatted as context with source labels
4. **LLM Generation** — Groq's LLaMA 3.1 8B Instant model generates an answer grounded strictly in the retrieved context
5. **Response** — The answer is returned along with source file names and similarity scores

---

##  How Endee is Used

[Endee](https://github.com/endee-io/endee) is the core vector database powering all semantic search in StudyMind.

### Setup
Endee is compiled from source with AVX2 acceleration and run as a local server on port `7070`:

```bash
git clone https://github.com/endee-io/endee
cd endee
CC=clang CXX=clang++ ./install.sh --release --avx2
./run.sh
```

### Index Creation
```python
from endee import Endee, Precision

client = Endee()
client.set_base_url("http://localhost:7070/api/v1")

client.create_index(
    name="studymind_index",
    dimension=384,        # matches all-MiniLM-L6-v2 output
    space_type="cosine",  # cosine similarity for semantic search
    precision=Precision.INT8  # quantized for memory efficiency
)
```

### Upserting Document Chunks
```python
index = client.get_index(name="studymind_index")

items = [
    {
        "id": chunk_id,            # deterministic MD5 hash
        "vector": embedding,       # 384-dim float list
        "meta": {
            "text": chunk_text,    # raw chunk for LLM context
            "source": filename,    # provenance tracking
            "chunk_id": i
        }
    }
    for i, (chunk_text, embedding) in enumerate(zip(chunks, vectors))
]

index.upsert(items)  # batch upsert in groups of 100
```

### Querying for Semantic Search
```python
query_vec = embedder.encode([query])[0].tolist()
results = index.query(vector=query_vec, top_k=5)

for r in results:
    print(r["meta"]["text"])    # retrieved passage
    print(r["score"])           # cosine similarity score
```

Endee handles all vector indexing and retrieval using an HNSW-based index with INT8 quantization, making it fast and memory-efficient for local deployment.

---

##  Project Structure

```
studymind/
├── backend/
│   ├── main.py           # FastAPI app — routes for upload, ask, summary, docs
│   ├── ingest.py         # Document parsing, chunking, embedding, Endee upsert
│   ├── retriever.py      # Semantic search via Endee query
│   ├── study_chain.py    # RAG chain — context builder + Groq LLM caller
│   └── .env              # API keys (GROQ_API_KEY, ENDEE_HOST)
├── frontend/
│   └── index.html        # Single-page UI (vanilla HTML/CSS/JS)
└── endee/                # Endee vector DB (cloned & compiled from source)
    ├── install.sh
    ├── run.sh
    └── build/ndd-avx2
```

---

##  Setup & Execution

### Prerequisites

- Python 3.10+
- `clang` / `clang++` (version 14+)
- `cmake`, `build-essential`, `libssl-dev`, `libcurl4-openssl-dev`
- A [Groq API key](https://console.groq.com/) (free tier available)

### 1. Clone and Build Endee

```bash
git clone https://github.com/endee-io/endee
cd endee
CC=clang CXX=clang++ ./install.sh --release --avx2
```

### 2. Start the Endee Server

```bash
cd endee
NDD_PORT=7070 NDD_AUTH_TOKEN="" NDD_NUM_THREADS=2 ./build/ndd-avx2
```

Verify it's running:
```bash
curl http://localhost:7070/api/v1/index/list
# Expected: {"indexes":[...]}
```

### 3. Install Python Dependencies

```bash
pip install fastapi uvicorn python-multipart sentence-transformers \
            pypdf groq python-dotenv endee requests
```

### 4. Configure Environment Variables

Create `backend/.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
ENDEE_HOST=http://localhost:7070
ENDEE_API_KEY=
```

### 5. Start the Backend

```bash
cd studymind/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 6. Open the UI

Navigate to `http://localhost:8000` in your browser.

> **For Google Colab:** Use [pyngrok](https://github.com/alexdlaird/pyngrok) to expose the local server publicly:
> ```python
> from pyngrok import ngrok
> public_url = ngrok.connect(8000)
> print(public_url)
> ```

---

##  Usage

1. **Upload** a `.pdf` or `.txt` document using the upload panel
2. **Ask a question** — e.g., *"What are the key concepts in chapter 3?"*
3. **Summarize a topic** — switch to Summarize mode and enter a topic
4. The system retrieves relevant passages from Endee and generates an answer with source attribution

---

##  Tech Stack

| Component | Technology |
|---|---|
| Vector Database | [Endee](https://github.com/endee-io/endee) |
| Embedding Model | `all-MiniLM-L6-v2` (SentenceTransformers) |
| LLM | LLaMA 3.1 8B Instant via [Groq](https://groq.com/) |
| Backend | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| PDF Parsing | `pypdf` |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Deployment (dev) | Google Colab + ngrok |

---

##  Notes

- The Endee index (`studymind_index`) persists across server restarts as long as the `./data` directory is preserved
- The document list is maintained in-memory; restarting the backend clears the document registry (vectors remain in Endee)
- For production use, replace ngrok with a proper deployment (e.g., Railway, Render, or a VPS)
- The frontend communicates with the backend over a configurable `API` base URL injected at startup

---

