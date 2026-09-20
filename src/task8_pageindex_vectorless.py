"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
Không có API key thì dùng tìm kiếm theo mục/điều trên Markdown local.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
CACHE_PATH = Path(__file__).parent.parent / "pageindex_doc_ids.json"
API_TIMEOUT_SECONDS = 15


def _load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_cache(mapping: dict[str, str]) -> None:
    CACHE_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")


def _call_with_timeout(func, *args, timeout: int = API_TIMEOUT_SECONDS, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY.strip():
        print("PAGEINDEX_API_KEY missing; skip remote upload, use local vectorless fallback.")
        return

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache = _load_cache()
    for path in sorted(LEGAL_DIR.glob("*")):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        if path.name in cache:
            continue
        try:
            response = _call_with_timeout(client.submit_document, str(path))
            doc_id = response.get("doc_id")
            if doc_id:
                cache[path.name] = str(doc_id)
                print(f"Uploaded {path.name} -> {doc_id}")
        except (Exception, FuturesTimeout) as error:
            print(f"Skip upload {path.name}: {error}")
    _save_cache(cache)


def _local_vectorless_search(query: str, top_k: int) -> list[dict]:
    """Tìm theo mục/điều trên Markdown khi không gọi được PageIndex API."""
    tokens = [token for token in re.findall(r"[0-9A-Za-zÀ-ỹ/%.-]+", query.lower()) if len(token) > 1]
    if not tokens:
        return []

    scored: list[tuple[float, dict]] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        sections = re.split(r"\n(?=Điều |\s*# )", content)
        doc_type = "legal" if "legal" in path.parts else "news"
        for index, section in enumerate(sections):
            text = section.strip()
            if len(text) < 80:
                continue
            lowered = text.lower()
            hits = sum(1 for token in tokens if token in lowered)
            if hits <= 0:
                continue
            score = hits / len(tokens)
            scored.append(
                (
                    score,
                    {
                        "id": f"pageindex::{path.relative_to(STANDARDIZED_DIR).as_posix()}::section-{index}",
                        "content": text[:1200],
                        "score": float(score),
                        "metadata": {
                            "source": path.name,
                            "title": path.stem,
                            "doc_type": doc_type,
                            "url": None,
                            "chunk_index": index,
                        },
                        "retrieval_method": "pageindex",
                    },
                )
            )

    scored.sort(key=lambda item: item[0], reverse=True)
    results = []
    seen = set()
    for _, item in scored:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        results.append(item)
        if len(results) >= top_k:
            break
    return results


def _remote_pageindex_search(query: str, top_k: int) -> list[dict]:
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache = _load_cache()
    if not cache:
        upload_documents()
        cache = _load_cache()

    results = []
    for source, doc_id in cache.items():
        try:
            submitted = _call_with_timeout(client.submit_query, doc_id, query)
            retrieval_id = submitted.get("retrieval_id")
            if not retrieval_id:
                continue
            payload = _call_with_timeout(client.get_retrieval, retrieval_id)
        except (Exception, FuturesTimeout):
            continue

        nodes = payload.get("nodes") or payload.get("results") or payload.get("retrieved_nodes") or []
        if isinstance(payload.get("content"), str) and not nodes:
            nodes = [{"content": payload["content"], "score": 1.0}]
        for index, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            content = str(node.get("content") or node.get("text") or "").strip()
            if not content:
                continue
            score = node.get("score")
            if not isinstance(score, (int, float)):
                score = max(0.0, 1.0 - index / max(top_k, 1))
            results.append(
                {
                    "id": f"pageindex::{source}::{index}",
                    "content": content,
                    "score": float(score),
                    "metadata": {
                        "source": source,
                        "title": Path(source).stem,
                        "doc_type": "legal",
                        "url": None,
                        "chunk_index": index,
                    },
                    "retrieval_method": "pageindex",
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if top_k <= 0 or not query.strip():
        return []
    if PAGEINDEX_API_KEY.strip():
        try:
            remote = _remote_pageindex_search(query, top_k)
            if remote:
                return remote
        except Exception:
            return []
    return _local_vectorless_search(query, top_k)


if __name__ == "__main__":
    upload_documents()
    for item in pageindex_search("Nghị định 349/2026 hiệu lực thi hành", top_k=3):
        print(f"{item['score']:.3f} | {item['id']}")
