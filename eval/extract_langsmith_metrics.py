import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langsmith import Client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.common import setup_logging, write_json

load_dotenv()


class LangSmithExtractor:
    def __init__(self, project_name: str, api_key: str):
        self.client = Client(api_key=api_key)
        self.project = project_name

    def get_runs(self, run_type: str = "chain", limit: int = 500) -> list[Any]:
        safe_limit = max(1, min(limit, 100))
        runs = list(self.client.list_runs(project_name=self.project, run_type=run_type, limit=safe_limit))
        filtered = [r for r in runs if "controller_run" in (getattr(r, "name", "") or "").lower()]
        return filtered

    def extract_turn_count(self, run: Any) -> int:
        child_runs = getattr(run, "child_runs", None) or []
        count = 0
        for c in child_runs:
            name = (getattr(c, "name", "") or "").lower()
            if "retriever_search" in name:
                count += 1
        return count if count > 0 else 1

    def extract_metrics_from_runs(self, runs: list[Any]) -> pd.DataFrame:
        records = []
        for run in runs:
            start = getattr(run, "start_time", None)
            end = getattr(run, "end_time", None)
            lat_ms = 0.0
            if start and end:
                lat_ms = (end - start).total_seconds() * 1000.0
            n_turns = self.extract_turn_count(run)
            outputs = getattr(run, "outputs", {}) or {}
            final_answer = ""
            if isinstance(outputs, dict):
                final_answer = str(outputs.get("output") or outputs.get("answer") or "")
            inputs = getattr(run, "inputs", {}) or {}
            question = str(inputs.get("query") or inputs.get("question") or "")
            records.append(
                {
                    "run_id": str(getattr(run, "id", "")),
                    "question": question,
                    "final_answer": final_answer,
                    "total_latency_ms": lat_ms,
                    "n_turns": n_turns,
                    "was_early_stop": n_turns < 3 and bool(final_answer),
                    "was_over_run": n_turns >= 3 and not bool(final_answer),
                    "embedding_calls": 0,
                    "retrieval_hit_rate": 0.0,
                }
            )
        return pd.DataFrame(records)


def compute_stopping_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "avg_turns": 0.0,
            "pct_1_turn": 0.0,
            "pct_2_turns": 0.0,
            "pct_3_turns": 0.0,
            "early_stop_rate": 0.0,
            "over_run_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    return {
        "avg_turns": float(df["n_turns"].mean()),
        "pct_1_turn": float((df["n_turns"] == 1).mean() * 100.0),
        "pct_2_turns": float((df["n_turns"] == 2).mean() * 100.0),
        "pct_3_turns": float((df["n_turns"] >= 3).mean() * 100.0),
        "early_stop_rate": float(df["was_early_stop"].mean() * 100.0),
        "over_run_rate": float(df["was_over_run"].mean() * 100.0),
        "avg_latency_ms": float(df["total_latency_ms"].mean()),
        "p95_latency_ms": float(df["total_latency_ms"].quantile(0.95)),
    }


def _plot_turn_distribution(df: pd.DataFrame, out_path: str) -> None:
    plt.rcParams["font.family"] = "serif"
    counts = [(df["n_turns"] == 1).mean() * 100.0, (df["n_turns"] == 2).mean() * 100.0, (df["n_turns"] >= 3).mean() * 100.0]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.bar(["Turn 1", "Turn 2", "Turn 3+"], counts)
    ax.set_ylabel("% queries")
    ax.set_title("Agentic turn distribution")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Axis 4 metrics from LangSmith.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", default="eval/agentic_stopping_stats.json")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    setup_logging("extract_langsmith_metrics")

    api_key = os.getenv("LANGSMITH_API_KEY", "")
    if not api_key:
        raise SystemExit("LANGSMITH_API_KEY is required.")
    extractor = LangSmithExtractor(project_name=args.project, api_key=api_key)
    runs = extractor.get_runs(limit=args.limit)
    df = extractor.extract_metrics_from_runs(runs)
    stats = compute_stopping_stats(df)
    write_json(args.output, stats)
    _plot_turn_distribution(df, "eval/turn_distribution.png")
    print(pd.DataFrame([stats]).to_string(index=False))


if __name__ == "__main__":
    main()

