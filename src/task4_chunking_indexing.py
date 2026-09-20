"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

from src.contracts import validate_document


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"
EMBED_BATCH_SIZE = 32

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DIM = 384
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")

COLLECTION_NAME = "rag_documents"

_sentence_model = None
_hash_vectorizer = None
_hash_fallback_warned = False
HASH_EMBEDDING_DIM = 384


def _title_from_markdown(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title:
                return title
    return fallback


def _url_from_markdown(content: str) -> str | None:
    match = re.search(r"https?://[^\s)>\]]+", content[:4000])
    return match.group(0).rstrip(".,;") if match else None


def _local_st_model_exists(name: str) -> bool:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + name.replace("/", "--")
    return (cache / slug).exists()


def _sentence_transformer():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer

        _sentence_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    return _sentence_model


def _hash_embeddings(texts: list[str]) -> list[list[float]]:
    global _hash_vectorizer
    if _hash_vectorizer is None:
        from sklearn.feature_extraction.text import HashingVectorizer

        _hash_vectorizer = HashingVectorizer(
            n_features=HASH_EMBEDDING_DIM,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )
    return _hash_vectorizer.transform(texts).toarray().tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts bằng đúng một provider; Task 5 phải gọi hàm này."""
    if not texts:
        return []

    provider = EMBEDDING_PROVIDER.lower().strip()
    if provider in {"sentence_transformers", "local", "huggingface"}:
        if _local_st_model_exists(EMBEDDING_MODEL):
            model = _sentence_transformer()
            vectors: list[list[float]] = []
            for start in range(0, len(texts), EMBED_BATCH_SIZE):
                batch = texts[start : start + EMBED_BATCH_SIZE]
                encoded = model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=len(texts) > EMBED_BATCH_SIZE,
                )
                vectors.extend(encoded.tolist())
            return vectors
        global _hash_fallback_warned
        if not _hash_fallback_warned:
            print(
                f"Embedding model {EMBEDDING_MODEL} is not cached locally; "
                "using hashing embeddings so indexing can finish offline."
            )
            _hash_fallback_warned = True
        return _hash_embeddings(texts)

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_name = EMBEDDING_MODEL or "text-embedding-3-small"
        vectors = []
        for start in range(0, len(texts), 100):
            batch = texts[start : start + 100]
            response = client.embeddings.create(model=model_name, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    if provider == "gemini":
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model_name = EMBEDDING_MODEL or "gemini-embedding-001"
        vectors = []
        for text in texts:
            response = client.models.embed_content(model=model_name, contents=text)
            vectors.append(list(response.embeddings[0].values))
        return vectors

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in path.parts else "news"
        relative_id = path.relative_to(STANDARDIZED_DIR).as_posix()
        document = {
            "id": relative_id,
            "content": content,
            "metadata": {
                "source": path.name,
                "title": _title_from_markdown(content, path.stem),
                "doc_type": doc_type,
                "url": _url_from_markdown(content),
            },
        }
        validate_document(document)
        documents.append(document)
    return documents


def _split_with_separators(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separator = separators[0] if separators else ""
    rest = separators[1:] if separators else []
    if separator == "":
        step = max(chunk_size - chunk_overlap, 1)
        return [text[start : start + chunk_size] for start in range(0, len(text), step)]

    merged: list[str] = []
    current = ""
    for piece in text.split(separator):
        candidate = piece if not current else f"{current}{separator}{piece}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            merged.append(current)
        if len(piece) > chunk_size:
            merged.extend(_split_with_separators(piece, rest, chunk_size, chunk_overlap))
            current = ""
        else:
            current = piece
    if current:
        merged.append(current)

    if chunk_overlap <= 0 or len(merged) <= 1:
        return merged

    overlapped = [merged[0]]
    for chunk in merged[1:]:
        prev = overlapped[-1]
        prefix = prev[-chunk_overlap:]
        if chunk.startswith(prefix):
            overlapped.append(chunk)
            continue
        candidate = prefix + chunk
        overlapped.append(chunk if len(candidate) > int(chunk_size * 1.1) else candidate)
    return overlapped


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    chunks = []
    for document in documents:
        pieces = [
            text.strip()
            for text in _split_with_separators(
                document["content"],
                ["\n\n", "\n", ". ", " ", ""],
                CHUNK_SIZE,
                CHUNK_OVERLAP,
            )
            if text.strip()
        ]
        for index, text in enumerate(pieces):
            chunk = {
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            }
            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    embedded = []
    for chunk, vector in zip(chunks, vectors):
        item = dict(chunk)
        item["embedding"] = vector
        embedded.append(item)
    return embedded


def _chroma_metadata(metadata: dict) -> dict:
    url = metadata.get("url")
    return {
        "source": str(metadata["source"]),
        "title": str(metadata["title"]),
        "doc_type": str(metadata["doc_type"]),
        "url": url if isinstance(url, str) and url.strip() else "",
        "chunk_index": int(metadata["chunk_index"]),
    }


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    collection = get_collection()
    batch_size = 200
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[_chroma_metadata(chunk["metadata"]) for chunk in batch],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks from {len(documents)} documents")


if __name__ == "__main__":
    run_pipeline()
