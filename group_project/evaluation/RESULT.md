# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-21 |
| Framework and version              | RAGAS 0.4.3 đã khai báo trong `pyproject.toml`; 4 metric đo bằng lexical proxy cùng tên (không gọi LLM-as-judge vì không có API key) |
| Evaluator model                    | Token-overlap + RAGAS-style context precision@k. Cùng công thức cho A và B. |
| Generator model                    | Cùng extractive generator `_extractive_answer` (fallback khi thiếu LLM key); citation `[Source: file]` |
| Embedding model                    | HashingVectorizer 384-d L2 khi thiếu `OPENAI_API_KEY`; cùng `embed_texts()` cho index và query |
| Corpus version/commit              | 8 legal + 8 news đã chuẩn hóa; 4278 chunks (recursive 500/50) |
| Golden dataset size                | 18 |
| `top_k`                            | 5 |
| Fallback threshold and calibration | `SCORE_THRESHOLD=0.33`. A/B này tắt fallback (`score_threshold=0`) để delta chỉ phản ánh RRF. |

## Configurations

- **Config A — dense-only:** `semantic_search(query, top_k=5)` trên Chroma cosine.
- **Config B — hybrid + RRF:** `retrieve(..., score_threshold=0.0, use_reranking=True)` = dense + BM25, fuse RRF một lần (`k=60`).

Hai config dùng cùng golden dataset, generator extractive, evaluator lexical, prompt citation và `top_k=5`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |    1.000 |    1.000 |     0.000 |
| Answer relevance  |    0.713 |    0.757 |    +0.045 |
| Context recall    |    0.869 |    0.893 |    +0.024 |
| Context precision |    0.939 |    0.948 |    +0.009 |
| **Average**       |    0.880 |    0.900 |    +0.019 |

Nguồn: `python group_project/evaluation/run_ab.py` → `ab_run.json`. Faithfulness lexical = 1.0 vì câu extractive luôn lấy từ chunk; metric này không phân biệt A/B khi không có LLM-as-judge.

## A/B comparison

- Cấu hình tốt hơn: **Config B (hybrid + RRF)**
- Evidence: B thắng 8/18 case, A thắng 3, hòa 7. B cao hơn A về relevance (+0.045), recall (+0.024) và precision (+0.009). BM25 giúp câu hỏi có số hiệu văn bản / ngưỡng tiền (NĐ 68, TT 91, 100 triệu đồng).
- Trade-off về latency/cost: B chậm hơn ~20 ms/câu (9.6 ms → 29.8 ms) vì thêm BM25 trên 4278 chunks; chưa gọi LLM nên chi phí API = 0.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Việc thực hiện Chương trình theo Thông tư 34/2026/TT-BYT phải bảo đảm nguyên tắc gì về trùng lặp nguồn lực? | A | 1.00 | 0.26 | 0.46 | 0.70 | generation | Context đủ một phần nhưng extractive không gom được cụm “không trùng lặp / chi thường xuyên” |
|   2 | Công nghệ chiến lược khác công nghệ cao ở điểm nào theo Luật 133/2025/QH15? | B | 1.00 | 0.34 | 0.26 | 0.89 | retrieval | Hai định nghĩa nằm sát nhau; recursive 500/50 + RRF lấy thiếu đoạn vàng so với dense |
|   3 | Công nghệ cao được định nghĩa như thế nào trong Luật số 133/2025/QH15? | A | 1.00 | 0.44 | 0.46 | 0.68 | generation | Chunk đúng file nhưng câu trả lời extractive cắt ngắn, overlap với expected_answer thấp |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Giữ Config B (hybrid + RRF) trên production; không quay về dense-only | B +0.019 average, thắng 8 và hòa 7 trên 18 case; latency +20 ms không đụng LLM | Giữ identifier/legal-code queries | Chạy lại `python group_project/evaluation/run_ab.py`; delta average B−A phải ≥ 0 |
|        2 | Chunk pháp luật theo Điều/khoản, giữ số hiệu trong metadata | Case định nghĩa CNC và “chiến lược”: evidence có trong file nhưng recursive 500/50 cắt rời hai khoản | Tăng context recall trên câu so sánh khái niệm | So 2 câu Luật 133 trước/sau; recall từng câu phải ≥ 0.7 |
|        3 | Khi có LLM key: thay extractive bằng `call_llm` + RAGAS LLM-as-judge | Faithfulness lexical = 1.0 không phân biệt; câu TT-34 context đủ mà relevance vẫn 0.26 | Faithfulness/relevance sát rubric hơn | Gắn `OPENAI_API_KEY` (hoặc Gemini/Anthropic), chạy lại 18 case, điền 4 metric RAGAS thật |

Không nhận bonus HyDE/cross-encoder: chưa có baseline + delta metric + latency/cost trên thí nghiệm đó.

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Không chạy | Hybrid + RRF (Config B) | n/a | n/a | Chưa đủ điều kiện nhận điểm bonus |
