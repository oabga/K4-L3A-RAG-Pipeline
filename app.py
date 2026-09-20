import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="RAG Pháp luật Việt Nam",
    page_icon="",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list, retrieval_source: str) -> None:
    st.caption(f"Phương thức truy xuất: `{retrieval_source or 'none'}`")
    if not sources:
        return
    for index, item in enumerate(sources, 1):
        metadata = item.get("metadata") or {}
        title = metadata.get("title") or "Không có tiêu đề"
        source = metadata.get("source") or ""
        method = item.get("retrieval_method") or ""
        score = item.get("score")
        score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
        url = metadata.get("url")
        header = f"{index}. {title} · `{source}` · {method} · score={score_text}"
        with st.expander(header, expanded=index == 1):
            if url:
                st.markdown(f"[Mở nguồn]({url})")
            st.write(item.get("content") or "")


with st.sidebar:
    st.title("RAG Chatbot")
    st.caption(
        "Đấu thầu (NĐ 349/2026), công nghệ cao (Luật 133/2025/QH15) "
        "và chăm sóc sức khỏe dân số (TT 34/2026/TT-BYT)."
    )
    top_k = st.slider("Số chunks", 3, 10, 5)

st.title("Hỏi đáp pháp luật có citation")
st.caption(
    "Câu trả lời chỉ dựa trên corpus đã thu thập. Nguồn hiển thị từ `sources` "
    "của `generate_with_citation`. Câu ngoài domain sẽ từ chối xác minh."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources") or [],
                message.get("retrieval_source") or "none",
            )

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và sinh câu trả lời..."):
            result = generate_with_citation(query, top_k)
        answer = result["answer"]
        sources = result["sources"]
        retrieval_source = result["retrieval_source"]
        st.markdown(answer)
        render_sources(sources, retrieval_source)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
        }
    )
