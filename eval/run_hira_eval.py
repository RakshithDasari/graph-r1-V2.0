import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controller import Controller
from agent.retriever import Retriever
from eval.common import append_jsonl, confirm_api_calls, load_jsonl, setup_logging, write_json
from eval.metrics import exact_match, f1_score, score_dataset_results


def _cooldown_seconds_from_error(error_text: str | None) -> float:
    if not error_text:
        return 0.0
    lower = error_text.lower()
    if "429" not in lower and "rate limit" not in lower:
        return 0.0

    # OpenRouter errors often include X-RateLimit-Reset epoch-ms header.
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


class HIRAEvaluator:
    def __init__(self, index_dir: str, langsmith_project: str):
        self.index_dir = index_dir
        self.langsmith_project = langsmith_project
        self.retriever = Retriever(index_dir)
        self.controller = Controller(max_turns=3)

    def query_with_timeout(self, question: str, timeout_seconds: int = 30) -> dict[str, Any]:
        start = time.perf_counter()

        def _run() -> dict[str, Any]:
            return self.controller.run_with_stats(question, self.retriever)

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_run)
        try:
            result = future.result(timeout=timeout_seconds)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return {
                "answer": str(result.get("answer", "") or ""),
                "turns": int(result.get("turns", self.controller.max_turns)),
                "latency_ms": latency_ms,
                "error": None,
            }
        except FuturesTimeoutError:
            future.cancel()
            latency_ms = (time.perf_counter() - start) * 1000.0
            return {
                "answer": "",
                "turns": self.controller.max_turns,
                "latency_ms": latency_ms,
                "error": "timeout",
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return {
                "answer": "",
                "turns": self.controller.max_turns,
                "latency_ms": latency_ms,
                "error": str(exc),
            }
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def run_dataset(
        self,
        dataset_path: str,
        output_path: str,
        resume: bool = True,
        test_mode: bool = False,
    ) -> None:
        rows = load_jsonl(dataset_path)
        if test_mode:
            rows = rows[:5]

        done_ids: set[str] = set()
        if resume and Path(output_path).exists():
            done_ids = {str(r.get("id", "")) for r in load_jsonl(output_path)}

        pending = [r for r in rows if str(r.get("id", "")) not in done_ids]
        start_all = time.perf_counter()
        scored: list[dict[str, Any]] = []
        existing = load_jsonl(output_path) if Path(output_path).exists() else []
        scored.extend(existing)

        for sample in pending:
            before = time.perf_counter()
            result = self.query_with_timeout(sample.get("question", ""))
            record = {
                "id": sample.get("id", ""),
                "dataset": sample.get("dataset", ""),
                "question": sample.get("question", ""),
                "ground_truth": sample.get("answer", ""),
                "prediction": result["answer"],
                "turns": result["turns"],
                "latency_ms": result["latency_ms"],
                "error": result["error"],
            }
            record["em"] = exact_match(record["prediction"], record["ground_truth"])
            record["f1"] = f1_score(record["prediction"], record["ground_truth"])
            append_jsonl(output_path, record)
            scored.append(record)

            elapsed = time.perf_counter() - before
            min_spacing = 8.0
            sleep_for = max(0.0, min_spacing - elapsed)
            cooldown = _cooldown_seconds_from_error(record.get("error"))
            if cooldown > sleep_for:
                sleep_for = cooldown
            if sleep_for > 0:
                print(f"[RateLimit] Sleeping {sleep_for:.2f}s")
                time.sleep(sleep_for)

            em_so_far = (sum(int(x.get("em", 0)) for x in scored) / len(scored)) * 100.0
            avg_latency_s = (sum(float(x.get("latency_ms", 0.0)) for x in scored) / len(scored)) / 1000.0
            print(f"{len(scored)}/{len(rows)} | EM so far: {em_so_far:.2f}% | avg latency: {avg_latency_s:.2f}s")

        duration = time.perf_counter() - start_all
        print(f"Finished in {duration:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HIRA eval over a sampled dataset.")
    parser.add_argument("--index-dir", default="artifacts")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--langsmith-project", default="hira-eval")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    log = setup_logging("run_hira_eval")
    rows = load_jsonl(args.dataset)
    if args.test:
        rows = rows[:5]
    confirm_api_calls(len(rows), yes=args.yes)

    evaluator = HIRAEvaluator(index_dir=args.index_dir, langsmith_project=args.langsmith_project)
    try:
        evaluator.run_dataset(
            dataset_path=args.dataset,
            output_path=args.output,
            resume=args.resume,
            test_mode=args.test,
        )
    except KeyboardInterrupt:
        print("Interrupted. Resume with --resume.")

    outputs = load_jsonl(args.output)
    metrics = score_dataset_results(
        [
            {
                "id": r.get("id", ""),
                "prediction": r.get("prediction", ""),
                "ground_truth": r.get("ground_truth", ""),
            }
            for r in outputs
        ]
    )
    avg_turns = sum(float(r.get("turns", 0)) for r in outputs) / len(outputs) if outputs else 0.0
    avg_latency_ms = sum(float(r.get("latency_ms", 0.0)) for r in outputs) / len(outputs) if outputs else 0.0
    timeout_rate = (sum(1 for r in outputs if r.get("error") == "timeout") / len(outputs) * 100.0) if outputs else 0.0
    error_rate = (sum(1 for r in outputs if r.get("error")) / len(outputs) * 100.0) if outputs else 0.0
    summary = {
        "exact_match": metrics["exact_match"],
        "f1": metrics["f1"],
        "n": metrics["n"],
        "avg_turns": avg_turns,
        "avg_latency_ms": avg_latency_ms,
        "timeout_rate_pct": timeout_rate,
        "error_rate_pct": error_rate,
    }
    summary_path = Path(args.output).with_name(f"summary_{Path(args.output).stem.replace('results_', '')}.json")
    write_json(summary_path, summary)
    csv_path = summary_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for key, value in summary.items():
            f.write(f"{key},{value}\n")
    print(json.dumps(summary, indent=2))
    log.info("Summary saved to %s and %s", summary_path.as_posix(), csv_path.as_posix())


if __name__ == "__main__":
    main()

