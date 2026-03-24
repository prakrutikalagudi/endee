from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

from ingest import ingest_document, list_documents, delete_document
from study_chain import answer_question, generate_summary

app = FastAPI(title="StudyMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")


# ---------------- UI ----------------
@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ---------------- HEALTH ----------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------- UPLOAD ----------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["text/plain", "application/pdf"]:
            raise HTTPException(400, "Only .txt and .pdf supported")

        content = await file.read()

        result = ingest_document(content, file.filename, file.content_type)

        return {"message": "uploaded", "chunks": result.get("chunks_stored", 0)}

    except Exception as e:
        print("UPLOAD ERROR:", str(e))  #  shows real error in logs
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- DOCUMENTS ----------------
@app.get("/documents")
def docs():
    return {"documents": list_documents()}


@app.delete("/documents/{filename}")
def delete(filename: str):
    delete_document(filename)
    return {"message": "deleted"}


# ---------------- ASK (RAG + GROQ) ----------------
@app.get("/ask")
def ask(query: str, top_k: int = 5):
    return answer_question(query, top_k)


# ---------------- SUMMARY (GROQ) ----------------
@app.get("/summary")
def summary(query: str):
    return generate_summary(query)


# ---------------- FIX ICON ERROR ----------------
@app.get("/favicon.ico")
async def favicon():
    return {}
