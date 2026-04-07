import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import faiss
import numpy as np
from openai import OpenAI

from agent.controller import Controller
from agent.retriever import Retriever
from eval.common import append_jsonl, confirm_api_calls, load_jsonl, setup_logging, write_json
from eval.metrics import exact_match, f1_score, score_dataset_results
from eval.rate_limiter import global_gemini_limiter
from graph.builder import chunk_document
from graph.encoder import GeminiEncoder


def _cooldown_seconds_from_error(error_text: str | None) -> float:
    if not error_text:
        return 0.0
    lower = error_text.lower()
    if "429" not in lower and "rate limit" not in lower:
        return 0.0

    match = re.search(r"'X-RateLimit-Reset':\s*'?(\d+)'?", error_text)
    if match:
        try:
            reset_ms = int(match.group(1))
            now_ms = int(time.time() * 1000)
            wait = max(0.0, (reset_ms - now_ms) / 1000.0)
            return min(max(wait + 0.5, 8.0), 30.0)
        except Exception:
            pass

    return 10.0


class FlatRAGBaseline:
    def __init__(self, docs_dir: str, index_path: str):
        self.docs_dir = Path(docs_dir)
        self.index_path = Path(index_path)
        self.meta_path = self.index_path.with_suffix(".json")
        self.encoder = GeminiEncoder()
        self.qwen = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_retries=0,
        )
        self.index: faiss.Index | None = None
        self.chunks: list[str] = []

    def build(self) -> None:
        texts: list[str] = []
        for txt_file in sorted(self.docs_dir.glob("**/*.txt")):
            chunks, _ = chunk_document(str(txt_file), chunk_size=500, overlap=100)
            texts.extend(chunks)
        if not texts:
            texts = ["No source text found."]
        embs = []
        for text in texts:
            with global_gemini_limiter:
                embs.append(self.encoder.encode(text))
        matrix = np.stack(embs).astype(np.float32)
        dim = matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(matrix)
        self.chunks = texts
        faiss.write_index(self.index, str(self.index_path))
        write_json(self.meta_path, {"chunks": texts})

    def _ensure_loaded(self) -> None:
        if self.index is None:
            self.index = faiss.read_index(str(self.index_path))
            self.chunks = []
            if self.meta_path.exists():
                self.chunks = json.loads(self.meta_path.read_text(encoding="utf-8")).get("chunks", [])

    def query(self, question: str) -> dict[str, Any]:
        self._ensure_loaded()
        start = time.perf_counter()
        try:
            with global_gemini_limiter:
                q = self.encoder.encode(question).reshape(1, -1).astype(np.float32)
            assert self.index is not None
            _, idx = self.index.search(q, 5)
            context = "\n".join(self.chunks[i] for i in idx[0] if 0 <= i < len(self.chunks))
            # Keep generation cheap and deterministic.
            resp = self.qwen.chat.completions.create(
                model="nvidia/nemotron-3-super-120b-a12b:free",
                temperature=0.0,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": "Answer using provided context only."},
                    {"role": "user", "content": f"Question: {question}\nContext:\n{context}"},
                ],
            )
            answer = (resp.choices[0].message.content or "").strip()
            return {"answer": answer, "turns": 1, "latency_ms": (time.perf_counter() - start) * 1000.0, "error": None}
        except Exception as exc:
            return {"answer": "", "turns": 1, "latency_ms": (time.perf_counter() - start) * 1000.0, "error": str(exc)}


class FixedTurnHIRA:
    def __init__(self, index_dir: str):
        self.retriever = Retriever(index_dir)
        self.controller = Controller(max_turns=1)

    def query(self, question: str, n_turns: int = 1) -> dict[str, Any]:
        self.controller.max_turns = n_turns
        start = time.perf_counter()
        try:
            with global_gemini_limiter:
                answer = self.controller.run(question, self.retriever)
            return {"answer": answer, "turns": n_turns, "latency_ms": (time.perf_counter() - start) * 1000.0, "error": None}
        except Exception as exc:
            return {"answer": "", "turns": n_turns, "latency_ms": (time.perf_counter() - start) * 1000.0, "error": str(exc)}


def _run_dataset(runner, dataset: str, output: str, resume: bool, test: bool) -> None:
    rows = load_jsonl(dataset)
    if test:
        rows = rows[:5]
    done = {str(r.get("id", "")) for r in load_jsonl(output)} if resume and Path(output).exists() else set()
    pending = [r for r in rows if str(r.get("id", "")) not in done]
    for row in pending:
        result = runner.query(row.get("question", ""))
        out = {
            "id": row.get("id", ""),
            "dataset": row.get("dataset", ""),
            "question": row.get("question", ""),
            "ground_truth": row.get("answer", ""),
            "prediction": result["answer"],
            "turns": result["turns"],
            "latency_ms": result["latency_ms"],
            "error": result["error"],
        }
        out["em"] = exact_match(out["prediction"], out["ground_truth"])
        out["f1"] = f1_score(out["prediction"], out["ground_truth"])
        append_jsonl(output, out)
        sleep_for = _cooldown_seconds_from_error(out.get("error"))
        if sleep_for <= 0:
            sleep_for = 8.0
        print(f"[RateLimit] Sleeping {sleep_for:.2f}s")
        time.sleep(sleep_for)

    final_rows = load_jsonl(output)
    summary = score_dataset_results(
        [{"id": r["id"], "prediction": r["prediction"], "ground_truth": r["ground_truth"]} for r in final_rows]
    )
    summary["avg_turns"] = sum(float(r.get("turns", 0)) for r in final_rows) / len(final_rows) if final_rows else 0.0
    summary["avg_latency_ms"] = (
        sum(float(r.get("latency_ms", 0.0)) for r in final_rows) / len(final_rows) if final_rows else 0.0
    )
    write_json(Path(output).with_name(f"summary_{Path(output).stem.replace('results_', '')}.json"), summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FlatRAG or FixedTurn baseline.")
    parser.add_argument("--baseline", choices=["flat_rag", "fixed_turn"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--docs-dir", default="data/sample")
    parser.add_argument("--index-path", default="eval/flatrag.index")
    parser.add_argument("--index-dir", default="artifacts")
    args = parser.parse_args()
    setup_logging("baselines")

    rows = load_jsonl(args.dataset)
    if args.test:
        rows = rows[:5]
    estimated_calls = len(rows) if args.baseline == "fixed_turn" else len(rows) + 1
    confirm_api_calls(estimated_calls, yes=args.yes)

    if args.baseline == "flat_rag":
        runner = FlatRAGBaseline(docs_dir=args.docs_dir, index_path=args.index_path)
        if not Path(args.index_path).exists():
            runner.build()
    else:
        runner = FixedTurnHIRA(index_dir=args.index_dir)

    _run_dataset(runner, dataset=args.dataset, output=args.output, resume=args.resume, test=args.test)
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()

