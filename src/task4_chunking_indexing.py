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

import os
import re
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
# - 500 ký tự (~120-150 token tiếng Việt): đủ cho một khoản/điểm của văn bản
#   pháp luật, giữ retrieval chính xác và context gửi LLM gọn.
# - overlap 50 (10%): tránh cắt mất ý ở ranh giới chunk.
# - recursive: ưu tiên tách theo đoạn, dòng, câu rồi mới tới từ.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Nhóm dùng OpenAI làm provider embedding mặc định (text-embedding-3-small, 1536 chiều).
# Đổi provider hoặc model thì phải xóa chroma_db/ và index lại.
DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
    "sentence_transformers": "BAAI/bge-m3",
}
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER") or "openai"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODELS.get(
    EMBEDDING_PROVIDER, ""
)
EMBEDDING_DIM = 1536  # dimension của text-embedding-3-small.

COLLECTION_NAME = "rag_documents"

EMBED_BATCH_SIZE = 32
INDEX_BATCH_SIZE = 500

_local_model = None


def _embed_sentence_transformers(texts: list[str]) -> list[list[float]]:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = _local_model.encode(
        texts, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True
    )
    return vectors.tolist()


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    response = OpenAI(max_retries=5).embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
    return [item.values for item in response.embeddings]


_EMBEDDERS = {
    "sentence_transformers": _embed_sentence_transformers,
    "openai": _embed_openai,
    "gemini": _embed_gemini,
}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách text bằng provider trong EMBEDDING_PROVIDER."""
    if not texts:
        return []
    try:
        embedder = _EMBEDDERS[EMBEDDING_PROVIDER]
    except KeyError:
        raise ValueError(
            f"EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r} không hợp lệ; "
            f"chọn một trong {sorted(_EMBEDDERS)}"
        ) from None
    return embedder(texts)


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _news_header(text: str) -> tuple[str | None, str | None]:
    """Lấy title và URL từ header Markdown do Task 3 tạo cho bài viết."""
    title = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    url = re.search(r"^\*\*Source:\*\*\s*(\S+)", text, flags=re.MULTILINE)
    return (
        title.group(1).strip() if title else None,
        url.group(1).strip() if url else None,
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in path.relative_to(STANDARDIZED_DIR).parts else "news"
        title, url = _news_header(content) if doc_type == "news" else (None, None)
        documents.append({
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": content,
            "metadata": {
                "source": path.name,
                "title": title or path.stem,
                "doc_type": doc_type,
                "url": url,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        texts = [t for t in splitter.split_text(document["content"]) if t.strip()]
        for index, text in enumerate(texts):
            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]
        vectors = embed_texts([chunk["content"] for chunk in batch])
        for chunk, vector in zip(batch, vectors):
            chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    collection = get_collection()
    for start in range(0, len(chunks), INDEX_BATCH_SIZE):
        batch = chunks[start:start + INDEX_BATCH_SIZE]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            # Chroma không nhận giá trị None; url thiếu sẽ được Task 5 điền lại.
            metadatas=[
                {k: v for k, v in chunk["metadata"].items() if v is not None}
                for chunk in batch
            ],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    print(f"Loaded {len(documents)} documents -> {len(chunks)} chunks")
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
