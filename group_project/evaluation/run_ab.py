"""A/B retrieval eval: Config A dense-only vs Config B hybrid+RRF.

Same golden dataset, extractive generator, lexical evaluator, top_k.
Fallback is disabled (score_threshold=0) so the delta isolates RRF.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import SAFE_REFUSAL, _extractive_answer, _tokens


GOLDEN_PATH = Path(__file__).with_name("golden_dataset.json")
OUT_PATH = Path(__file__).with_name("ab_run.json")
TOP_K = 5
STANDARDIZED = ROOT / "data" / "standardized"


def _chunk_relevant(content: str, expected_context: str) -> bool:
    gold = _tokens(expected_context)
    chunk = _tokens(content)
    if not gold or not chunk:
        return False
    overlap = len(gold & chunk)
    return overlap >= 8 or overlap / len(gold) >= 0.12


def context_recall(chunks: list[dict], expected_context: str) -> float:
    gold = _tokens(expected_context)
    if not gold:
        return 0.0
    blob = _tokens(" ".join(item["content"] for item in chunks))
    return len(gold & blob) / len(gold)


def context_precision(chunks: list[dict], expected_context: str) -> float:
    if not chunks:
        return 0.0
    flags = [_chunk_relevant(item["content"], expected_context) for item in chunks]
    relevant_n = sum(flags)
    if relevant_n == 0:
        return 0.0
    running = 0
    score = 0.0
    for rank, flag in enumerate(flags, 1):
        running += int(flag)
        score += (running / rank) * int(flag)
    return score / relevant_n


def faithfulness(answer: str, chunks: list[dict]) -> float:
    cleaned = re.sub(r"\[Source:.*?\]", " ", answer)
    if SAFE_REFUSAL in answer:
        return 1.0 if not chunks else 0.0
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[\.!?…])\s+", cleaned)
        if len(part.strip()) > 20
    ]
    if not sentences:
        sentences = [cleaned]
    ctx = _tokens(" ".join(item["content"] for item in chunks))
    if not ctx:
        return 0.0
    scores = []
    for sentence in sentences:
        tokens = _tokens(sentence)
        if not tokens:
            continue
        scores.append(len(tokens & ctx) / len(tokens))
    return mean(scores) if scores else 0.0


def answer_relevance(question: str, answer: str, expected_answer: str) -> float:
    if SAFE_REFUSAL in answer:
        return 0.0
    cleaned = re.sub(r"\[Source:.*?\]", " ", answer)
    expected = _tokens(expected_answer)
    generated = _tokens(cleaned)
    asked = _tokens(question)
    gold_recall = (len(expected & generated) / len(expected)) if expected else 0.0
    question_hit = (len(asked & generated) / len(asked)) if asked else 0.0
    return 0.7 * gold_recall + 0.3 * question_hit


def generate(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return SAFE_REFUSAL
    return _extractive_answer(query, chunks)


def retrieve_a(query: str) -> list[dict]:
    return semantic_search(query, top_k=TOP_K)


def retrieve_b(query: str) -> list[dict]:
    return retrieve(query, top_k=TOP_K, score_threshold=0.0, use_reranking=True)


def corpus_contains(expected_context: str) -> bool:
    needle = re.sub(r"\s+", " ", expected_context).strip().lower()
    if len(needle) < 40:
        return False
    window = needle[:80]
    for path in STANDARDIZED.rglob("*.md"):
        hay = re.sub(r"\s+", " ", path.read_text(encoding="utf-8")).lower()
        if window in hay or needle[:60] in hay:
            return True
        tokens = _tokens(expected_context)
        if tokens and len(tokens & _tokens(hay)) / len(tokens) >= 0.8:
            return True
    return False


def classify(row: dict, in_corpus: bool) -> tuple[str, str]:
    recall = row["context_recall"]
    precision = row["context_precision"]
    relevance = row["answer_relevance"]
    faith = row["faithfulness"]
    if not in_corpus:
        return "data", "expected_context không tìm thấy đủ trong corpus đã chuẩn hóa"
    if recall < 0.45 or precision < 0.35:
        return (
            "retrieval",
            f"context recall={recall:.2f}, precision={precision:.2f}: đoạn vàng có trong corpus nhưng top-{TOP_K} không đưa đủ evidence",
        )
    if relevance < 0.45 or faith < 0.55:
        return (
            "generation",
            f"context ổn (recall={recall:.2f}) nhưng answer relevance={relevance:.2f}, faithfulness={faith:.2f}",
        )
    return "none", "không phải failure nặng"


def run_config(name: str, retriever, cases: list[dict]) -> dict:
    rows = []
    elapsed = []
    for case in cases:
        question = case["question"]
        started = time.perf_counter()
        chunks = retriever(question)
        answer = generate(question, chunks)
        elapsed.append(time.perf_counter() - started)
        metrics = {
            "faithfulness": round(faithfulness(answer, chunks), 4),
            "answer_relevance": round(
                answer_relevance(question, answer, case["expected_answer"]), 4
            ),
            "context_recall": round(context_recall(chunks, case["expected_context"]), 4),
            "context_precision": round(
                context_precision(chunks, case["expected_context"]), 4
            ),
        }
        metrics["average"] = round(mean(metrics.values()), 4)
        rows.append(
            {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "source": item["metadata"]["source"],
                        "method": item.get("retrieval_method"),
                        "score": round(float(item["score"]), 4),
                    }
                    for item in chunks
                ],
                **metrics,
            }
        )
    keys = ["faithfulness", "answer_relevance", "context_recall", "context_precision", "average"]
    overall = {key: round(mean(row[key] for row in rows), 4) for key in keys}
    overall["latency_s_mean"] = round(mean(elapsed), 4)
    return {"name": name, "overall": overall, "cases": rows}


def main() -> None:
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    retrieve_a(cases[0]["question"])
    retrieve_b(cases[0]["question"])
    config_a = run_config("A_dense", retrieve_a, cases)
    config_b = run_config("B_hybrid_rrf", retrieve_b, cases)
    in_corpus = [corpus_contains(case["expected_context"]) for case in cases]

    paired = []
    for index, (row_a, row_b, present) in enumerate(
        zip(config_a["cases"], config_b["cases"], in_corpus)
    ):
        worse = row_a if row_a["average"] <= row_b["average"] else row_b
        config_name = "A" if worse is row_a else "B"
        stage, cause = classify(worse, present)
        paired.append(
            {
                "index": index,
                "question": row_a["question"],
                "in_corpus": present,
                "A": {
                    k: row_a[k]
                    for k in (
                        "faithfulness",
                        "answer_relevance",
                        "context_recall",
                        "context_precision",
                        "average",
                    )
                },
                "B": {
                    k: row_b[k]
                    for k in (
                        "faithfulness",
                        "answer_relevance",
                        "context_recall",
                        "context_precision",
                        "average",
                    )
                },
                "worse_config": config_name,
                "failure_stage": stage,
                "root_cause": cause,
                "sources_A": row_a["sources"],
                "sources_B": row_b["sources"],
                "answer_A": row_a["answer"][:400],
                "answer_B": row_b["answer"][:400],
            }
        )

    worst = sorted(
        paired, key=lambda item: min(item["A"]["average"], item["B"]["average"])
    )[:3]
    payload = {
        "top_k": TOP_K,
        "fallback": "disabled (score_threshold=0)",
        "A": {"overall": config_a["overall"]},
        "B": {"overall": config_b["overall"]},
        "delta_B_minus_A": {
            key: round(config_b["overall"][key] - config_a["overall"][key], 4)
            for key in config_a["overall"]
        },
        "cases": paired,
        "worst": worst,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "A": config_a["overall"],
            "B": config_b["overall"],
            "delta": payload["delta_B_minus_A"],
        },
        indent=2,
    ))
    print("\nWorst 3:")
    for item in worst:
        print(f"- [{item['worse_config']}/{item['failure_stage']}] {item['question'][:80]}")
        print(f"  A={item['A']['average']} B={item['B']['average']} :: {item['root_cause']}")


if __name__ == "__main__":
    main()
