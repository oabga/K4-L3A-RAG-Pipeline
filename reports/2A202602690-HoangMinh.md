# Individual contribution report

## Thông tin

- Họ và tên: Trang Phước Hoàng Minh
- Mã học viên: 2A202602690
- Nhóm: K4-L3A
- Repository/branch: https://github.com/oabga/K4-L3A-RAG-Pipeline / `hoang-minh` (`5de3d34`)

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Corpus pháp luật + tin | Thu thập 3 PDF (NĐ 349/2026, Luật 133/2025/QH15, TT 34/2026/TT-BYT) và 5 bài Thư viện Pháp luật; convert Markdown; giữ URL/title/date | `src/task1_collect_legal_docs.py`, `src/task2_crawl_news.py`, `src/task3_convert_markdown.py`, `src/corpus_utils.py`, `data/landing/`, `data/standardized/` · `5de3d34` | Done |
| Chunk + index | Recursive 500/50, Chroma cosine, `embed_texts()` dùng chung; fallback HashingVectorizer 384-d khi MiniLM chưa cache | `src/task4_chunking_indexing.py` · `5de3d34` | Done |
| Hybrid retrieval | Dense Chroma + BM25 cùng schema `SearchResult`; RRF `sum(1/(60+rank))` đúng một lần; fallback PageIndex/local theo cosine dense gốc | `src/task5_semantic_search.py` … `src/task9_retrieval_pipeline.py` · `5de3d34` | Done |
| Generation + UI | Citation `[Source: file]`, safe refusal ngoài domain, extractive khi thiếu LLM key; Streamlit hiện answer, nguồn, method, score | `src/task10_generation.py`, `app.py` · `5de3d34` | Done |
| Evaluation | 15 golden Q&A; 4 metric lexical/RAGAS-style; A/B dense-only vs hybrid+RRF; điền báo cáo | `group_project/evaluation/golden_dataset.json`, `run_ab.py`, `ab_run.json`, `RESULT.md` · `5de3d34` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Fallback so sánh `SCORE_THRESHOLD=0.33` với cosine dense gốc, không dùng điểm RRF.  
   **Lý do/evidence:** Hiệu chỉnh 2026-09-20 trên corpus này: in-domain dense top-1 min=0.367 mean=0.460; out-of-domain max=0.301 mean=0.253 (`src/task9_retrieval_pipeline.py`). Query in-domain (“NĐ 349 hiệu lực khi nào?”) giữ hybrid; query ngoài domain (“Cách nấu phở…”) rơi xuống threshold.  
   **Trade-off:** Threshold gắn embedding hashing hiện tại; đổi MiniLM phải calibrate lại. RRF không còn dùng được làm ngưỡng vì thang khác cosine.

2. **Quyết định:** Production dùng hybrid+RRF; A/B tắt fallback (`score_threshold=0`) để delta chỉ đo retrieval.  
   **Lý do/evidence:** `python group_project/evaluation/run_ab.py` trên 15 case: B thắng 14/15, average 0.591 → 0.794; answer relevance +0.396, context recall +0.349. Hashing-dense trượt identifier (`349/2026`, `GRDP`, `100 triệu`); BM25 kéo đúng `article_05.md` / NĐ-349.  
   **Trade-off:** Latency ~6 ms → ~32 ms/câu sau warmup, không gọi thêm LLM. Hybrid không sửa lỗi extractive (câu “100 triệu” vẫn trích “01 tỷ”).

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `pytest -q` (`tests/test_contracts.py`, `tests/test_acceptance.py`); golden 15 câu; in-domain vs OOD trong `task9`/`task10`; A/B `run_ab.py`.
- Kết quả trước/sau nếu có: dense-only average 0.591; hybrid+RRF 0.794 (delta +0.204). Case GRDP: A recall 0.16 / precision 0.00 → B recall 1.00 nhờ BM25.
- Lỗi đã phát hiện và cách xử lý: Thư viện Pháp luật chặn crawler (Cloudflare) → lưu bản công khai rồi convert; MiniLM chưa cache → hashing embeddings cùng dim 384; thiếu LLM key → extractive + metric lexical, ghi rõ trong `RESULT.md`.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: Hashing embeddings + chunk recursive 500/50 làm dense lẫn văn bản gần nghĩa (định nghĩa CNC vs TT-BYT) và cắt rời nguyên tắc TT-34; generator extractive copy câu nên faithfulness lexical luôn 1.0, không phân biệt A/B.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: chunk pháp luật theo Điều/khoản (giữ số hiệu trong metadata) rồi đo lại 3 worst case (GRDP, trùng lặp nguồn lực TT-34, công nghệ chiến lược).

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 21/09/2026
- Tên thành viên: Trang Phước Hoàng Minh
