"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi


CORPUS: list[dict] = []


def _ensure_corpus() -> list[dict]:
    if CORPUS:
        return CORPUS
    from src.task4_chunking_indexing import chunk_documents, load_documents

    CORPUS.extend(chunk_documents(load_documents()))
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    tokenized = [item["content"].lower().split() or [""] for item in corpus]
    if not tokenized:
        tokenized = [[""]]
    return BM25Okapi(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip():
        return []

    corpus = _ensure_corpus()
    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    scores = bm25.get_scores(query.lower().split())
    indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]

    results = []
    seen = set()
    for index in indices:
        score = float(scores[index])
        if score < 0:
            continue
        item = corpus[int(index)]
        item_id = item["id"]
        if item_id in seen:
            continue
        seen.add(item_id)
        results.append(
            {
                "id": item_id,
                "content": item["content"],
                "score": score,
                "metadata": item["metadata"],
                "retrieval_method": "bm25",
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in lexical_search("Nghị định 349/2026 100 triệu đồng", top_k=3):
        print(f"{result['score']:.3f} | {result['id']} | {result['content'][:120]}")
