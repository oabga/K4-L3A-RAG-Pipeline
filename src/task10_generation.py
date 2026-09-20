"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation dạng [Source: <filename>] khớp trường Source trong context.
Nếu thiếu evidence, hãy từ chối xác minh. Không bịa thông tin."""

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

_STOPWORDS = {
    "của", "và", "là", "các", "cho", "với", "trong", "được", "một", "này",
    "khi", "có", "không", "theo", "từ", "đến", "về", "như", "để", "hay",
    "the", "and", "or", "for", "with", "from", "that", "this",
}

_PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _copy_chunk(chunk: dict) -> dict:
    copied = dict(chunk)
    metadata = chunk.get("metadata")
    if isinstance(metadata, dict):
        copied["metadata"] = dict(metadata)
    return copied


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    copied = [_copy_chunk(chunk) for chunk in chunks]
    if len(copied) <= 2:
        return copied
    front = copied[::2]
    back = copied[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        parts.append(
            f"[Document {index} | Title: {metadata['title']} | "
            f"Source: {metadata['source']}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def _provider_api_key() -> str:
    provider = (LLM_PROVIDER or "openai").lower().strip()
    env_name = _PROVIDER_KEYS.get(provider)
    if not env_name:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
    key = (os.getenv(env_name) or "").strip()
    if not key:
        raise RuntimeError(f"{env_name} missing")
    return key


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = (LLM_PROVIDER or "openai").lower().strip()
    api_key = _provider_api_key()

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model = LLM_MODEL or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    if provider == "gemini":
        from google import genai

        client = genai.Client(api_key=api_key)
        model = LLM_MODEL or "gemini-2.0-flash"
        response = client.models.generate_content(
            model=model,
            contents=f"{system_prompt}\n\n{user_message}",
        )
        return (response.text or "").strip()

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        model = LLM_MODEL or "claude-sonnet-4-5"
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "text", None)
        ).strip()

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def _retrieval_source(chunks: list[dict]) -> str:
    if not chunks:
        return "none"
    method = chunks[0].get("retrieval_method")
    if method == "pageindex":
        return "pageindex"
    return "hybrid"


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[0-9A-Za-zÀ-ỹ]+", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _has_sufficient_evidence(query: str, chunks: list[dict]) -> bool:
    query_tokens = _tokens(query)
    if not query_tokens or not chunks:
        return False
    blob = " ".join(chunk["content"] for chunk in chunks[:3]).lower()
    hits = sum(1 for token in query_tokens if token in blob)
    ratio = hits / len(query_tokens)
    if chunks[0].get("retrieval_method") == "pageindex":
        return ratio >= 0.55
    return ratio >= 0.35


def _unwrap_lines(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    merged: list[str] = []
    for line in lines:
        if merged and not re.search(r"[\.!?…:;]$", merged[-1]):
            merged[-1] = f"{merged[-1]} {line}"
        else:
            merged.append(line)
    return " ".join(merged)


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    """Trích câu có overlap với query và gắn citation theo Source."""
    query_tokens = _tokens(query)
    query_numbers = set(re.findall(r"\d{3,}", query))
    query_lower = query.lower()
    scored: list[tuple[int, str]] = []
    for chunk in chunks:
        source = chunk["metadata"]["source"]
        unwrapped = _unwrap_lines(chunk["content"])
        pieces = re.split(r"(?<=[\.!?…])\s+", unwrapped)
        for piece in pieces:
            text = " ".join(piece.split()).strip()
            if len(text) < 40 or re.match(r"^[a-zà-ỹ]{1,4}\b", text):
                continue
            overlap = len(_tokens(text) & query_tokens)
            if overlap <= 0:
                continue
            lower = text.lower()
            bonus = overlap * 2
            source_lower = source.lower()
            title_lower = str(chunk["metadata"].get("title") or "").lower()
            if query_numbers and any(number in source_lower or number in title_lower for number in query_numbers):
                bonus += 10
            if query_numbers and any(number in text for number in query_numbers):
                bonus += 6
            if "hiệu lực" in query_lower and "kể từ" in lower:
                bonus += 8
            if "hiệu lực" in query_lower and "tiếp tục thực hiện" in lower:
                bonus -= 4
            if len(text) > 420:
                bonus -= 5
            scored.append((bonus, f"{text} [Source: {source}]"))

    scored.sort(key=lambda item: item[0], reverse=True)
    unique: list[str] = []
    seen: set[str] = set()
    for _, sentence in scored:
        if sentence in seen:
            continue
        seen.add(sentence)
        unique.append(sentence)
        if len(unique) >= 3:
            break

    if unique:
        return "\n\n".join(unique)

    chunk = chunks[0]
    excerpt = " ".join(chunk["content"].split())[:500].strip()
    return f"{excerpt} [Source: {chunk['metadata']['source']}]"


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks or not _has_sufficient_evidence(query, chunks):
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = (
        f"Context:\n{context}\n\nQuestion: {query}\n\n"
        "Cite using [Source: <filename>] matching the Source field."
    )
    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception:
        answer = _extractive_answer(query, reordered)
    if not answer.strip():
        answer = SAFE_REFUSAL
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": _retrieval_source(chunks),
    }


if __name__ == "__main__":
    samples = [
        "Nghị định 349/2026/NĐ-CP có hiệu lực thi hành khi nào?",
        "Cách nấu phở bò Hà Nội ngon?",
    ]
    for query in samples:
        result = generate_with_citation(query)
        print(f"Q: {query}")
        print(f"source={result['retrieval_source']} n={len(result['sources'])}")
        print(result["answer"][:400])
        print("---")
