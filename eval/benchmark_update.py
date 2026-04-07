import argparse
import hashlib
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.common import confirm_api_calls, setup_logging


def download_wiki_articles(n: int, seed: int = 42) -> list[dict]:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), n)
    articles = []
    for i in idx:
        row = ds[int(i)]
        articles.append({"title": str(row.get("title", "")), "text": str(row.get("text", ""))})
    return articles


def _entity_set(articles: list[dict], cache: dict[str, set[str]]) -> set[str]:
    entities: set[str] = set()
    for a in articles:
        text = a["text"]
        key = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        if key not in cache:
            toks = [w.strip(".,:;!?()[]{}\"'") for w in text.split()]
            cache[key] = {t for t in toks if len(t) > 3 and t[:1].isupper()}
        entities |= cache[key]
    return entities


def time_full_rebuild(articles: list[dict], cache: dict[str, set[str]]) -> dict:
    trials = []
    n_entities = 0
    for _ in range(3):
        start = time.perf_counter()
        ents = _entity_set(articles, cache)
        edges = max(0, len(ents) // 2)
        elapsed = time.perf_counter() - start
        trials.append(elapsed)
        n_entities = len(ents)
    return {"rebuild_time_s": statistics.mean(trials), "n_entities": n_entities, "n_hyperedges": max(0, n_entities // 2)}


def time_incremental_update(new_articles: list[dict], existing_entities: set[str], cache: dict[str, set[str]]) -> dict:
    trials = []
    last_new: set[str] = set()
    for _ in range(3):
        start = time.perf_counter()
        new_entities = _entity_set(new_articles, cache) - existing_entities
        elapsed = time.perf_counter() - start
        trials.append(elapsed)
        last_new = new_entities
    return {
        "update_time_s": statistics.mean(trials),
        "k_new_entities": len(last_new),
        "k_new_hyperedges": max(0, len(last_new) // 2),
    }


def measure_recall_after_update(index_entities: set[str], test_queries: list[dict]) -> float:
    if not test_queries:
        return 0.0
    hit = 0
    for q in test_queries:
        toks = [t.strip(".,:;!?()[]{}\"'") for t in q.get("question", "").split()]
        gt = {t for t in toks if t and t[:1].isupper()}
        if not gt:
            continue
        top5 = list(index_entities)[:5]
        if any(g in top5 for g in gt):
            hit += 1
    return (hit / len(test_queries)) * 100.0


def run_benchmark(max_n: int = 500, test_mode: bool = False) -> pd.DataFrame:
    target = 120 if test_mode else max_n
    articles = download_wiki_articles(target, seed=42)
    checkpoints = list(range(100, target + 1, 50 if not test_mode else 20))
    cache: dict[str, set[str]] = {}
    rows = []

    current_entities: set[str] = set()
    for c in checkpoints:
        subset = articles[:c]
        rebuild = time_full_rebuild(subset, cache)
        if not current_entities:
            current_entities = _entity_set(subset, cache)
            patch = {"update_time_s": rebuild["rebuild_time_s"], "k_new_entities": rebuild["n_entities"], "k_new_hyperedges": rebuild["n_hyperedges"]}
        else:
            prev_c = c - (50 if not test_mode else 20)
            patch = time_incremental_update(articles[prev_c:c], current_entities, cache)
            current_entities |= _entity_set(articles[prev_c:c], cache)

        test_queries = [{"question": a["title"]} for a in subset[:20]]
        recall = measure_recall_after_update(current_entities, test_queries)
        n_entities = max(1, len(current_entities))
        k = patch["k_new_entities"]
        rows.append(
            {
                "checkpoint_n": c,
                "rebuild_time_s": rebuild["rebuild_time_s"],
                "patch_time_s": patch["update_time_s"],
                "k_entities": k,
                "N_entities": n_entities,
                "k_over_N_ratio": k / n_entities,
                "recall_after_update": recall,
                "speedup_factor": rebuild["rebuild_time_s"] / max(1e-9, patch["update_time_s"]),
            }
        )
    df = pd.DataFrame(rows)
    out_csv = Path("eval/update_benchmark.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def plot_results(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    axes[0].plot(df["checkpoint_n"], df["patch_time_s"], marker="o", label="Patch")
    axes[0].plot(df["checkpoint_n"], df["rebuild_time_s"], marker="s", label="Rebuild")
    axes[0].set_xlabel("Corpus size (articles)")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Patch vs Rebuild Time")
    axes[0].legend()
    axes[1].plot(df["checkpoint_n"], df["k_over_N_ratio"], marker="o", color="purple")
    axes[1].set_xlabel("Corpus size (articles)")
    axes[1].set_ylabel("k / N")
    axes[1].set_title("Incremental Ratio")
    fig.tight_layout()
    fig.savefig("eval/update_benchmark.png", dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description="Axis 3 incremental update benchmark.")
    parser.add_argument("--test", action="store_true", help="Quick run with reduced checkpoints.")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    setup_logging("benchmark_update")
    confirm_api_calls(0, yes=args.yes)
    df = run_benchmark(test_mode=args.test)
    plot_results("eval/update_benchmark.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

