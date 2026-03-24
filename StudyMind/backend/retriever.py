from ingest import _get_embedder, _get_endee_index

def semantic_search(query, top_k=5):
    index = _get_endee_index()
    embedder = _get_embedder()

    query_vec = embedder.encode([query])[0].tolist()

    results = index.query(vector=query_vec, top_k=top_k)

    output = []
    for r in results:
        meta = r.get("meta", {})   

        output.append({
            "text": meta.get("text", ""),
            "source": meta.get("source", ""),
            "similarity": r.get("score", 0)  
        })

    return output
