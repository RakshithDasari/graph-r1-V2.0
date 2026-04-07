import argparse
import collections
import random
import re
import string
from typing import Iterable

import numpy as np


ARTICLES = {"a", "an", "the"}


def normalize_answer(answer: str) -> str:
    text = answer.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(token for token in text.split() if token not in ARTICLES)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    common_tokens = sum(common.values())
    if common_tokens == 0:
        return 0.0
    precision = common_tokens / len(pred_tokens)
    recall = common_tokens / len(gt_tokens)
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0


def score_dataset_results(results: list[dict]) -> dict:
    em_per_sample = [exact_match(r["prediction"], r["ground_truth"]) for r in results]
    f1_per_sample = [f1_score(r["prediction"], r["ground_truth"]) for r in results]
    n = len(results)
    em = (sum(em_per_sample) / n) * 100 if n else 0.0
    f1 = (sum(f1_per_sample) / n) * 100 if n else 0.0
    return {
        "exact_match": em,
        "f1": f1,
        "n": n,
        "em_per_sample": em_per_sample,
        "f1_per_sample": f1_per_sample,
    }


def compute_confidence_interval(scores: list[float], confidence: float = 0.95) -> tuple[float, float]:
    if not scores:
        return (0.0, 0.0)
    n_bootstrap = 1000
    rng = random.Random(42)
    arr = np.array(scores, dtype=float)
    n = len(arr)
    boots = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(float(arr[idx].mean()))
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(boots, alpha))
    upper = float(np.quantile(boots, 1.0 - alpha))
    return (lower, upper)


def _run_self_test() -> None:
    tests = [
        ("Barack Obama", "Obama", 0, 0.6666666667),
        ("The Eiffel Tower", "eiffel tower", 1, 1.0),
        ("Paris", "London", 0, 0.0),
        ("an apple", "apple", 1, 1.0),
        ("", "", 1, 1.0),
    ]
    ok = True
    for i, (pred, gt, em_exp, f1_exp) in enumerate(tests, start=1):
        em_act = exact_match(pred, gt)
        f1_act = f1_score(pred, gt)
        em_pass = em_act == em_exp
        f1_pass = abs(f1_act - f1_exp) < 1e-6
        print(
            f"Test {i}: EM expected={em_exp}, actual={em_act} | "
            f"F1 expected={f1_exp:.6f}, actual={f1_act:.6f} | "
            f"{'PASS' if em_pass and f1_pass else 'FAIL'}"
        )
        ok = ok and em_pass and f1_pass
    if not ok:
        raise SystemExit("Self-test failed.")
    print("All 5 self-tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HIRA evaluation metrics module.")
    parser.add_argument("--test", action="store_true", help="Run self-test.")
    args = parser.parse_args()
    if args.test:
        _run_self_test()

