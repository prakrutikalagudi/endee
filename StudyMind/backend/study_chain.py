import os
from dotenv import load_dotenv
from retriever import semantic_search

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


# ---------------- LLM CALL ----------------
def _call_llm(prompt):
    if GROQ_API_KEY:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )

        return response.choices[0].message.content.strip()

    return None


# ---------------- CONTEXT ----------------
def _build_context(chunks):
    return "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    ])

# ---------------- ASK (RAG) ----------------
def answer_question(question, top_k=5):
    chunks = semantic_search(question, top_k)

    if not chunks:
        return {
            "answer": "No documents uploaded.",
            "sources": []
        }

    context = _build_context(chunks)

    prompt = f"""
You are an AI assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = _call_llm(prompt)

    if not answer:
        answer = chunks[0]["text"][:300]

    return {
        "answer": answer,
        "sources": [
            {
                "source": c["source"],
                "score": c["similarity"]   # fixed: was c["score"]
            } for c in chunks
        ]
    }


# ---------------- SUMMARY ----------------
def generate_summary(query):
    chunks = semantic_search(query, 5)

    if not chunks:
        return {
            "summary": "No documents uploaded."
        }

    context = _build_context(chunks)

    prompt = f"""
Summarize the following content clearly:

{context}

Summary:
"""

    summary = _call_llm(prompt)

    if not summary:
        summary = context[:500]

    return {
        "summary": summary
    }
