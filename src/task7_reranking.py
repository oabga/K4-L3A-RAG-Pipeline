"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.
"""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in items:
                items[item_id] = {
                    "id": item["id"],
                    "content": item["content"],
                    "metadata": dict(item.get("metadata") or {}),
                    "score": item["score"],
                    "retrieval_method": item.get("retrieval_method", "dense"),
                }

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    results = []
    for item_id in ranked_ids[: max(top_k, 0)]:
        result = dict(items[item_id])
        result["metadata"] = dict(items[item_id]["metadata"])
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)
    return results


if __name__ == "__main__":
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search

    query = "Gói thầu không quá 100 triệu đồng"
    dense = semantic_search(query, top_k=5)
    sparse = lexical_search(query, top_k=5)
    fused = rerank_rrf([dense, sparse], top_k=5, k=60)
    print(f"dense={len(dense)} bm25={len(sparse)} hybrid={len(fused)}")
    for item in fused[:3]:
        print(f"{item['score']:.4f} | {item['id']}")
