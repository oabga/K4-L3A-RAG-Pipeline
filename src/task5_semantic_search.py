"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip():
        return []

    query_vector = embed_texts([query])[0]
    try:
        response = get_collection().query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as error:
        message = str(error)
        if "dimension" in message.lower():
            raise RuntimeError(
                "ChromaDB lệch chiều embedding (collection cũ khác model hiện tại). "
                "Chạy lại: python -m src.task4_chunking_indexing"
            ) from error
        raise

    results = []
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        # Task 4 bỏ url=None khi lưu vào Chroma; điền lại để đúng contract.
        metadata = {**metadata}
        metadata.setdefault("url", None)
        results.append({
            "id": item_id,
            "content": content,
            "score": max(0.0, 1.0 - distance),
            "metadata": metadata,
            "retrieval_method": "dense",
        })
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)
