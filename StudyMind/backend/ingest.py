import re, io, os, hashlib
from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

ENDEE_HOST = os.getenv("ENDEE_HOST", "http://localhost:7070")
ENDEE_API_KEY = os.getenv("ENDEE_API_KEY", "")
ENDEE_INDEX = "studymind_index"

_embedder = None
_endee_client = None
_endee_index = None
_uploaded_docs = set()

def _get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading embedding model...")
        _embedder = SentenceTransformer(EMBED_MODEL)
        print("Model ready.")
    return _embedder

def _get_endee_index():
    global _endee_client, _endee_index

    if _endee_index is None:
        import requests
        from endee import Endee, Precision

        _endee_client = Endee(ENDEE_API_KEY) if ENDEE_API_KEY else Endee()
        _endee_client.set_base_url(f"{ENDEE_HOST}/api/v1")

        # list_indexes() returns raw strings, not objects
        try:
            resp = requests.get(f"{ENDEE_HOST}/api/v1/index/list")
            data = resp.json()
            existing_names = [idx["name"] for idx in data.get("indexes", [])]
        except Exception:
            existing_names = []

        if ENDEE_INDEX not in existing_names:
            _endee_client.create_index(
                name=ENDEE_INDEX,
                dimension=384,
                space_type="cosine",
                precision=Precision.INT8
            )

        _endee_index = _endee_client.get_index(name=ENDEE_INDEX)
        print("✅ Connected to Endee index")

    return _endee_index

def _extract_text(content, content_type):
    if content_type == "text/plain":
        return content.decode("utf-8", errors="ignore")

    if content_type == "application/pdf":
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    raise ValueError(f"Unsupported type: {content_type}")

def _chunk_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    chunks, start = [], 0

    while start < len(text):
        chunk = text[start:start+CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

def _chunk_id(filename, i):
    return hashlib.md5(f"{filename}__chunk_{i}".encode()).hexdigest()

def ingest_document(content, filename, content_type):
    text = _extract_text(content, content_type)

    if not text.strip():
        raise ValueError("Document is empty.")

    chunks = _chunk_text(text)

    embedder = _get_embedder()
    vectors = embedder.encode(chunks, show_progress_bar=True).tolist()

    index = _get_endee_index()

    items = []
    for i in range(len(chunks)):
        items.append({
            "id": _chunk_id(filename, i),
            "vector": vectors[i],
            "meta": {
                "text": chunks[i],
                "source": filename,
                "chunk_id": i
            }
        })

    # batch insert
    for i in range(0, len(items), 100):
        index.upsert(items[i:i+100])

    _uploaded_docs.add(filename)

    return {"chunks_stored": len(chunks)}

def list_documents():
    return sorted(_uploaded_docs)

def delete_document(filename):
    _uploaded_docs.discard(filename)
