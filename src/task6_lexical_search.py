"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


import re
import unicodedata

CORPUS: list[dict] = []

_TOKEN_PATTERN = re.compile(r"\w+")
_index = None
_index_corpus: list[dict] | None = None


def tokenize(text: str) -> list[str]:
    """Chuẩn hóa Unicode, hạ chữ thường và tách theo từ (bỏ dấu câu)."""
    return _TOKEN_PATTERN.findall(unicodedata.normalize("NFC", text).lower())


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(item["content"]) for item in corpus])


def _load_corpus() -> list[dict]:
    """Nạp CORPUS bằng đúng chunks (cùng ID) mà Task 4 đã index vào Chroma."""
    global CORPUS
    if not CORPUS:
        from .task4_chunking_indexing import chunk_documents, load_documents

        CORPUS = chunk_documents(load_documents())
    return CORPUS


def _get_index(corpus: list[dict]):
    """Cache index; chỉ build lại khi CORPUS bị thay đổi."""
    global _index, _index_corpus
    if _index is None or _index_corpus is not corpus:
        _index = build_bm25_index(corpus)
        _index_corpus = corpus
    return _index


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    import numpy as np

    tokens = tokenize(query)
    corpus = _load_corpus()
    if top_k <= 0 or not tokens or not corpus:
        return []

    bm25 = _get_index(corpus)
    scores = bm25.get_scores(tokens)
    results = []
    for index in np.argsort(scores)[::-1]:
        if len(results) >= top_k:
            break
        # Không lọc theo score > 0: IDF có thể bằng 0 (từ có trong đúng nửa corpus),
        # nên chỉ loại chunk không chứa từ nào của query.
        if not any(token in bm25.doc_freqs[index] for token in tokens):
            continue
        item = corpus[index]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": {**item["metadata"]},
            "retrieval_method": "bm25",
        })
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
