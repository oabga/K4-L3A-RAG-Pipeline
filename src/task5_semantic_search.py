"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def _from_chroma_metadata(metadata: dict) -> dict:
    url = metadata.get("url")
    if not url:
        url = None
    return {
        "source": str(metadata.get("source") or "unknown"),
        "title": str(metadata.get("title") or "untitled"),
        "doc_type": str(metadata.get("doc_type") or "legal"),
        "url": url,
        "chunk_index": int(metadata.get("chunk_index") or 0),
    }


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip():
        return []

    query_vector = embed_texts([query])[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = (response.get("ids") or [[]])[0]
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]

    results = []
    seen = set()
    for item_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        results.append(
            {
                "id": item_id,
                "content": content or "",
                "score": max(0.0, 1.0 - float(distance)),
                "metadata": _from_chroma_metadata(metadata or {}),
                "retrieval_method": "dense",
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("Nghị định 349/2026 đấu thầu 100 triệu đồng", top_k=3):
        print(f"{result['score']:.3f} | {result['id']} | {result['content'][:120]}")
