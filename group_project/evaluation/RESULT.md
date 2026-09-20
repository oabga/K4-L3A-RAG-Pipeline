# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20 |
| Framework and version              | RAGAS 0.4.3 đã cài; 4 metric đo bằng lexical proxy cùng tên (không gọi LLM-as-judge vì không có API key) |
| Evaluator model                    | Token-overlap + RAGAS-style context precision@k. Cùng công thức cho A và B. |
| Generator model                    | Cùng extractive generator `_extractive_answer` (fallback Streamlit khi thiếu LLM key); cùng citation `[Source: file]` |
| Embedding model                    | HashingVectorizer 384-d L2 (HF `paraphrase-multilingual-MiniLM-L12-v2` chưa cache được); cùng `embed_texts()` cho index và query |
| Corpus version/commit              | HEAD `6a2a2d4` + working tree 2026-09-20: 3 văn bản pháp luật + 5 bài Thư viện Pháp luật, 1131 chunks |
| Golden dataset size                | 15 (`group_project/evaluation/golden_dataset.json`) |
| `top_k`                            | 5 |
| Fallback threshold and calibration | A/B **tắt** PageIndex (`score_threshold=0`) để delta chỉ đo retrieval strategy. Pipeline production vẫn dùng `SCORE_THRESHOLD=0.33` trên cosine dense gốc. |

Script tái lập: `python group_project/evaluation/run_ab.py` → `ab_run.json`.

## Configurations

- **Config A — dense-only:** `semantic_search` Chroma cosine, không BM25, không RRF.
- **Config B — hybrid + RRF:** dense + BM25, fuse đúng một lần `sum(1/(60+rank))`.

Hai config dùng cùng 15 golden cases, cùng generator, cùng evaluator, cùng prompt extractive và `top_k=5`. Chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |    1.000 |    1.000 |     0.000 |
| Answer relevance  |    0.186 |    0.582 |    +0.396 |
| Context recall    |    0.439 |    0.788 |    +0.349 |
| Context precision |    0.737 |    0.807 |    +0.069 |
| **Average**       |    0.591 |    0.794 |    +0.204 |

Faithfulness = 1.0 ở cả hai vì generator extractive copy câu từ chunk; metric này không phân biệt A/B. Tín hiệu A/B nằm ở recall, precision và answer relevance.

## A/B comparison

- Cấu hình tốt hơn trên **trung bình:** Config B (hybrid + RRF). B thắng 14/15 case. Lợi lớn nhất là answer relevance (+0.396) và context recall (+0.349): BM25 kéo đúng số hiệu / định danh (`349/2026`, `100 triệu`, `GRDP`, `11%`) mà hashing-dense hay trượt sang Thông tư 34 hoặc Luật CNC.
- Evidence cụ thể: câu GRDP 2026–2030, A lấy 5 chunk TT-BYT/NĐ-349 (`recall=0.16`, `precision=0.00`); B đưa `article_05.md` vào top-5 (`recall=1.00`). Câu “100 triệu đồng”: A không lấy được NĐ-349; B có 2 chunk NĐ-349, recall 0.55 → 0.86.
- Case B **kém hơn A:** “Công nghệ cao được định nghĩa…”. Average A 0.633 vs B 0.625. Dense precision 1.0 nhưng nguồn A toàn TT-BYT (overlap từ vựng chung, lexical precision bị thổi); B trộn TT-BYT với Luật CNC nên recall định nghĩa vẫn 0.35. Không chọn B chỉ vì average: định nghĩa thuật ngữ gần nghĩa vẫn lỗi retrieval.
- Trade-off latency/cost: sau warmup, A ~6 ms/câu, B ~32 ms/câu (+26 ms, khoảng 5×) vì BM25 quét corpus. Không gọi thêm LLM. Chi phí tiền không đổi; đánh đổi chấp nhận được so với +0.35 recall. Hybrid không sửa được lỗi extractive (câu 100 triệu B vẫn trích “không quá 01 tỷ” thay vì “100 triệu”).

## Worst performers

Ba case có `min(avg_A, avg_B)` thấp nhất. Điểm trong bảng là config kém hơn.

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Mục tiêu GRDP TP.HCM 2026–2030 theo dự thảo Chương trình hành động NQ TW 3? | A dense-only | 1.00 | 0.11 | 0.16 | 0.00 | retrieval | Đoạn vàng nằm ở `article_05.md`. Hashing-dense trả TT-34 (chỉ số y tế) và NĐ-349; cosine top-1 chỉ 0.09. B khôi phục bằng BM25 (`GRDP`, `11%`) nhưng answer B vẫn trích tiêu đề bài, relevance 0.30 → lỗi generation còn lại |
|   2 | Chương trình theo TT 34/2026/TT-BYT phải bảo đảm nguyên tắc gì về trùng lặp nguồn lực? | A dense-only | 1.00 | 0.19 | 0.33 | 0.20 | retrieval | Nguyên tắc “không trùng lặp… chi thường xuyên” nằm sâu trong TT-34 dài. Dense lấy chunk TT-34 khác chủ đề + NĐ-349 + Luật CNC. B tăng precision 0.20→0.50 nhưng recall chỉ 0.33→0.37: chunk 500/50 cắt rời câu nguyên tắc |
|   3 | Công nghệ chiến lược khác công nghệ cao ở điểm nào theo Luật 133/2025/QH15? | A dense-only | 1.00 | 0.03 | 0.33 | 0.45 | retrieval | Hai định nghĩa gần nhau trong cùng luật. Dense đưa NĐ-349 lên trên Luật CNC (score 0.054 vs 0.050). B kéo được Luật CNC nhưng recall 0.35: chunk định nghĩa “công nghệ chiến lược” không vào top-5 |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Giữ Config B (hybrid + RRF) trên production; không quay về dense-only | B +0.35 recall, +0.40 relevance, thắng 14/15; latency +26 ms không đụng LLM | Giữ identifier/legal-code queries | Chạy lại `python group_project/evaluation/run_ab.py`; delta recall B−A phải ≥ 0.25 trên cùng 15 case |
|        2 | Chunk pháp luật theo Điều/khoản, giữ số hiệu trong metadata | Case 2–3 và “100 triệu”: evidence có trong file nhưng recursive 500/50 + RRF vẫn thiếu hoặc lẫn “01 tỷ” | Tăng context precision và giảm near-miss generation | So 3 câu: 100 triệu, định nghĩa CNC, nguyên tắc trùng lặp TT-34 trước/sau; recall từng câu phải ≥ 0.7 |
|        3 | Khi có LLM key: thay extractive bằng `call_llm` + RAGAS LLM-as-judge | Faithfulness lexical = 1.0 không phân biệt; câu GRDP/100 triệu context đủ mà câu trả lời vẫn lệch | Faithfulness/relevance sát rubric hơn | Gắn `OPENAI_API_KEY` (hoặc Gemini/Anthropic), chạy lại 15 case, điền 4 metric RAGAS thật; 3 worst phải đổi stage nếu retrieval đã ổn |

Không nhận bonus HyDE/cross-encoder: chưa có baseline + delta metric + latency/cost trên thí nghiệm đó.

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Không chạy | Hybrid + RRF (Config B) | n/a | n/a | Chưa đủ điều kiện nhận điểm bonus |
