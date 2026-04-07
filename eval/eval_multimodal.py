import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.retriever import Retriever
from eval.common import confirm_api_calls, load_jsonl, setup_logging, write_json
from eval.llm_judge import LLMJudge
from eval.rate_limiter import global_gemini_limiter
from graph.encoder import GeminiEncoder

load_dotenv()


def _download_image(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    suffix = ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(resp.content)
        return f.name


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multimodal retrieval and QA.")
    parser.add_argument("--dataset", default="data/eval/multimodal_50.jsonl")
    parser.add_argument("--index-dir", default="artifacts")
    parser.add_argument("--output", default="eval/multimodal_results.json")
    parser.add_argument("--summary-output", default="eval/multimodal_summary.json")
    parser.add_argument("--judge-model", default="nvidia/nemotron-3-super-120b-a12b:free")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    setup_logging("eval_multimodal")

    rows = load_jsonl(args.dataset)
    if args.test:
        rows = rows[:5]
    confirm_api_calls(len(rows) * 2, yes=args.yes)

    retriever = Retriever(args.index_dir)
    encoder = GeminiEncoder()
    judge = LLMJudge(model=args.judge_model)

    results = []
    for row in rows:
        with global_gemini_limiter:
            q_emb = encoder.encode(row["question"])
        image_path = _download_image(row["image_url"])
        try:
            with global_gemini_limiter:
                img_emb = encoder.encode(image_path)

            with_ctx = retriever.search(row["question"], top_k=5)
            image_node_in_top5 = any(e.get("name") == image_path for e in with_ctx.get("entities", []))
            sim = _cos(q_emb, img_emb)

            pred_with = with_ctx["facts"][0] if with_ctx["facts"] else ""
            without_ctx = retriever.search(row["question"], top_k=5)
            pred_without = without_ctx["facts"][0] if without_ctx["facts"] else ""

            judged_with = judge.judge_single(row["question"], pred_with, row["answer"])
            judged_without = judge.judge_single(row["question"], pred_without, row["answer"])
            results.append(
                {
                    "id": row["id"],
                    "image_node_in_top5": bool(image_node_in_top5),
                    "answer_correct_with_image": bool(judged_with["correct"]),
                    "answer_correct_without_image": bool(judged_without["correct"]),
                    "embedding_similarity": sim,
                }
            )
        finally:
            try:
                os.remove(image_path)
            except OSError:
                pass

    write_json(args.output, results)
    n = max(1, len(results))
    summary = {
        "image_node_recall_at_5": sum(1 for r in results if r["image_node_in_top5"]) / n * 100.0,
        "cross_modal_accuracy": sum(1 for r in results if r["answer_correct_with_image"]) / n * 100.0,
        "text_only_accuracy": sum(1 for r in results if r["answer_correct_without_image"]) / n * 100.0,
        "delta": (
            sum(1 for r in results if r["answer_correct_with_image"]) - sum(1 for r in results if r["answer_correct_without_image"])
        )
        / n
        * 100.0,
        "avg_embedding_similarity": float(np.mean([r["embedding_similarity"] for r in results])) if results else 0.0,
    }
    write_json(args.summary_output, summary)
    print(summary)


if __name__ == "__main__":
    main()

