import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation, llm_is_configured


load_dotenv()

st.set_page_config(
    page_title="LexRAG · Hỏi đáp pháp luật",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUESTIONS = [
    "Nghị định 349/2026/NĐ-CP có hiệu lực thi hành khi nào?",
    "Hộ kinh doanh doanh thu 500 triệu đồng có chịu thuế GTGT không?",
    "Công nghệ cao được định nghĩa thế nào trong Luật 133/2025/QH15?",
    "Cách nấu phở bò Hà Nội ngon?",
]

METHOD_LABELS = {
    "hybrid": "Hybrid + RRF",
    "dense": "Dense",
    "bm25": "BM25",
    "pageindex": "PageIndex",
    "none": "Không retrieve",
}


st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');

      html, body, [class*="st-"] {
        font-family: "Be Vietnam Pro", sans-serif;
      }
      .stApp {
        background:
          radial-gradient(1200px 500px at 10% -10%, #1e3a5f 0%, transparent 55%),
          linear-gradient(180deg, #0b1220 0%, #111827 42%, #0f172a 100%);
        color: #e5e7eb;
      }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stToolbar"] { display: none; }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1524 0%, #111c2e 100%);
        border-right: 1px solid rgba(201, 162, 39, 0.18);
      }
      [data-testid="stSidebar"] * { color: #e5e7eb; }
      .hero {
        border: 1px solid rgba(201, 162, 39, 0.28);
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 58, 95, 0.55));
        border-radius: 22px;
        padding: 22px 26px 18px;
        margin-bottom: 18px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
      }
      .kicker {
        color: #c9a227;
        letter-spacing: 0.16em;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 6px;
      }
      .hero h1 {
        font-family: "Source Serif 4", serif;
        font-size: 2.05rem;
        margin: 0 0 8px;
        color: #f8fafc;
      }
      .hero p { color: #cbd5e1; margin: 0; line-height: 1.55; }
      .chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
      .chip {
        background: rgba(201, 162, 39, 0.12);
        border: 1px solid rgba(201, 162, 39, 0.32);
        color: #f3e6b3;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 12px;
      }
      .status-ok, .status-warn {
        border-radius: 14px;
        padding: 10px 12px;
        font-size: 13px;
        line-height: 1.45;
      }
      .status-ok {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #bbf7d0;
      }
      .status-warn {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fde68a;
      }
      [data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 16px;
        padding: 8px 10px;
      }
      .source-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 10px 12px;
        margin-bottom: 8px;
      }
      .source-meta { color: #94a3b8; font-size: 12px; }
      .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(201, 162, 39, 0.28);
        background: rgba(15, 23, 42, 0.7);
        color: #f8fafc;
      }
      .stButton > button:hover {
        border-color: #c9a227;
        color: #c9a227;
      }
      [data-testid="stChatInput"] textarea {
        border-radius: 16px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

with st.sidebar:
    st.markdown("### ⚖️ LexRAG")
    st.caption("Chatbot RAG pháp luật Việt Nam · citation theo nguồn đã retrieve")

    st.divider()
    llm_ready = llm_is_configured()
    if llm_ready:
        st.markdown(
            '<div class="status-ok">LLM đọc từ file <code>.env</code>. '
            "Câu trả lời chỉ dựa trên chunk đã retrieve.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-warn">Chưa thấy API key trong <code>.env</code>. '
            "Đang dùng extractive. Restart app sau khi sửa <code>.env</code>.</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    top_k = st.slider("Số chunks retrieve", 3, 10, 5)
    if st.button("Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Corpus: NĐ 349/2026, Luật 133/2025/QH15, TT 34/2026/TT-BYT, "
        "NĐ 68/2026, TT 91/2026 và bài viết chính sách."
    )

incoming = (st.session_state.pending_query or "").strip()
if incoming:
    st.session_state.pending_query = ""
    st.session_state.messages.append({"role": "user", "content": incoming})
    with st.spinner("Đang retrieve rồi gửi context cho LLM..."):
        result = generate_with_citation(incoming, top_k)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
            "used_llm": bool(result.get("used_llm")),
        }
    )
    st.rerun()

st.markdown(
    """
    <div class="hero">
      <div class="kicker">Retrieval-grounded legal assistant</div>
      <h1>Hỏi đáp pháp luật có nguồn</h1>
      <p>Hệ thống retrieve trước, rồi mới sinh câu trả lời. LLM chỉ được dùng các đoạn đã tìm thấy — không bịa điều khoản.</p>
      <div class="chip-row">
        <span class="chip">Dense + BM25 + RRF</span>
        <span class="chip">Citation theo file nguồn</span>
        <span class="chip">Từ chối ngoài domain</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown("**Thử một câu hỏi**")
    cols = st.columns(2)
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        if cols[index % 2].button(question, use_container_width=True):
            st.session_state.pending_query = question
            st.rerun()


def render_sources(sources: list, retrieval_source: str, used_llm: bool | None = None) -> None:
    method = METHOD_LABELS.get(retrieval_source or "none", retrieval_source or "none")
    engine = "LLM grounded" if used_llm else "Extractive / không LLM"
    st.caption(f"Truy xuất: **{method}** · Sinh câu trả lời: **{engine}**")
    if not sources:
        return
    st.markdown("**Nguồn đã retrieve**")
    for index, item in enumerate(sources, 1):
        metadata = item.get("metadata") or {}
        title = metadata.get("title") or "Không có tiêu đề"
        source = metadata.get("source") or ""
        score = item.get("score")
        score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
        url = metadata.get("url")
        header = f"{index}. {title}"
        with st.expander(header, expanded=index == 1):
            st.markdown(
                f'<div class="source-meta">{source} · {item.get("retrieval_method") or ""} · score {score_text}</div>',
                unsafe_allow_html=True,
            )
            if url:
                st.markdown(f"[Mở nguồn gốc]({url})")
            st.write(item.get("content") or "")


for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "⚖️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources") or [],
                message.get("retrieval_source") or "none",
                message.get("used_llm"),
            )

query = st.chat_input("Hỏi về nghị định, thông tư, luật...")
if query:
    st.session_state.pending_query = query
    st.rerun()
