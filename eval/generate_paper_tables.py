import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.common import load_jsonl


def _read_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _metric_from_jsonl(path: str) -> dict[str, float]:
    rows = load_jsonl(path)
    if not rows:
        return {"em": 0.0, "f1": 0.0, "avg_turns": 0.0, "latency_s": 0.0}
    em = sum(float(r.get("em", 0.0)) for r in rows) / len(rows) * 100.0
    f1 = sum(float(r.get("f1", 0.0)) for r in rows) / len(rows) * 100.0
    turns = sum(float(r.get("turns", 0.0)) for r in rows) / len(rows)
    latency = sum(float(r.get("latency_ms", 0.0)) for r in rows) / len(rows) / 1000.0
    return {"em": em, "f1": f1, "avg_turns": turns, "latency_s": latency}


def _fmt(v: float) -> str:
    return f"{v:.2f}"


def _bold_if_best(values: list[float], idx: int) -> str:
    val = values[idx]
    if val == max(values):
        return f"\\textbf{{{_fmt(val)}}}"
    return _fmt(val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables and paper summary artifacts.")
    parser.add_argument("--eval-dir", default="eval")
    args = parser.parse_args()
    eval_dir = Path(args.eval_dir)

    summaries = {
        "hotpotqa": _read_json(str(eval_dir / "summary_hotpotqa.json")),
        "2wikimhqa": _read_json(str(eval_dir / "summary_2wikimhqa.json")),
        "musique": _read_json(str(eval_dir / "summary_musique.json")),
    }
    baselines = {
        "flatrag_hotpotqa": _metric_from_jsonl(str(eval_dir / "results_flatrag_hotpotqa.jsonl")),
        "fixed_hotpotqa": _metric_from_jsonl(str(eval_dir / "results_fixed_hotpotqa.jsonl")),
    }
    mm = _read_json(str(eval_dir / "multimodal_summary.json"))
    upd = pd.read_csv(eval_dir / "update_benchmark.csv") if (eval_dir / "update_benchmark.csv").exists() else pd.DataFrame()
    axis4 = _read_json(str(eval_dir / "agentic_stopping_stats.json"))

    hira_hotpot = summaries["hotpotqa"]
    hira_2wiki = summaries["2wikimhqa"]
    hira_musique = summaries["musique"]
    fr = baselines["flatrag_hotpotqa"]
    fx = baselines["fixed_hotpotqa"]

    hotpot_ems = [float(hira_hotpot.get("exact_match", 0.0)), fr["em"], fx["em"]]
    hotpot_f1s = [float(hira_hotpot.get("f1", 0.0)), fr["f1"], fx["f1"]]

    main_table = rf"""\begin{{table*}}[t]
\centering
\caption{{Main Results Across Datasets and Baselines}}
\begin{{tabular}}{{l|cccc|cccc|cccc}}
\hline
Model & \multicolumn{{4}}{{c|}}{{HotpotQA}} & \multicolumn{{4}}{{c|}}{{2Wiki}} & \multicolumn{{4}}{{c}}{{MuSiQue}} \\
& EM & F1 & Avg-Turns & Lat(s) & EM & F1 & Avg-Turns & Lat(s) & EM & F1 & Avg-Turns & Lat(s) \\
\hline
HIRA & {_bold_if_best(hotpot_ems, 0)} & {_bold_if_best(hotpot_f1s, 0)} & {_fmt(float(hira_hotpot.get('avg_turns',0.0)))} & {_fmt(float(hira_hotpot.get('avg_latency_ms',0.0))/1000.0)} & {_fmt(float(hira_2wiki.get('exact_match',0.0)))} & {_fmt(float(hira_2wiki.get('f1',0.0)))} & {_fmt(float(hira_2wiki.get('avg_turns',0.0)))} & {_fmt(float(hira_2wiki.get('avg_latency_ms',0.0))/1000.0)} & {_fmt(float(hira_musique.get('exact_match',0.0)))} & {_fmt(float(hira_musique.get('f1',0.0)))} & {_fmt(float(hira_musique.get('avg_turns',0.0)))} & {_fmt(float(hira_musique.get('avg_latency_ms',0.0))/1000.0)} \\
FlatRAG & {_bold_if_best(hotpot_ems, 1)} & {_bold_if_best(hotpot_f1s, 1)} & {_fmt(fr['avg_turns'])} & {_fmt(fr['latency_s'])} & - & - & - & - & - & - & - & - \\
FixedTurn-HIRA & {_bold_if_best(hotpot_ems, 2)} & {_bold_if_best(hotpot_f1s, 2)} & {_fmt(fx['avg_turns'])} & {_fmt(fx['latency_s'])} & - & - & - & - & - & - & - & - \\
\hline
\multicolumn{{13}}{{l}}{{Multimodal recall@5: {_fmt(float(mm.get('image_node_recall_at_5',0.0)))}\% \quad Update speedup: {_fmt(float(upd['speedup_factor'].max()) if not upd.empty else 0.0)}x}} \\
\hline
\end{{tabular}}
\end{{table*}}
"""

    ablation_table = rf"""\begin{{table}}[t]
\centering
\caption{{Ablation Study}}
\begin{{tabular}}{{lccc}}
\hline
Variant & EM (HotpotQA) & F1 (HotpotQA) & Avg-Turns \\
\hline
HIRA full & {_fmt(float(hira_hotpot.get('exact_match',0.0)))} & {_fmt(float(hira_hotpot.get('f1',0.0)))} & {_fmt(float(hira_hotpot.get('avg_turns',0.0)))} \\
HIRA no-multimodal & {_fmt(float(mm.get('text_only_accuracy',0.0)))} & - & {_fmt(float(hira_hotpot.get('avg_turns',0.0)))} \\
HIRA fixed-turn & {_fmt(fx['em'])} & {_fmt(fx['f1'])} & {_fmt(fx['avg_turns'])} \\
\hline
\end{{tabular}}
\end{{table}}
"""

    (Path("table_main_results.tex")).write_text(main_table, encoding="utf-8")
    (Path("table_ablation.tex")).write_text(ablation_table, encoding="utf-8")

    paper_numbers = {
        "hira_hotpotqa_em": float(hira_hotpot.get("exact_match", 0.0)),
        "hira_hotpotqa_f1": float(hira_hotpot.get("f1", 0.0)),
        "hira_2wikimhqa_em": float(hira_2wiki.get("exact_match", 0.0)),
        "hira_2wikimhqa_f1": float(hira_2wiki.get("f1", 0.0)),
        "hira_musique_em": float(hira_musique.get("exact_match", 0.0)),
        "hira_musique_f1": float(hira_musique.get("f1", 0.0)),
        "multimodal_recall_at_5": float(mm.get("image_node_recall_at_5", 0.0)),
        "update_speedup_factor_max": float(upd["speedup_factor"].max()) if not upd.empty else 0.0,
        "axis4_avg_turns": float(axis4.get("avg_turns", 0.0)),
    }
    (eval_dir / "paper_numbers.json").write_text(json.dumps(paper_numbers, indent=2), encoding="utf-8")

    summary_md = f"""# Evaluation Summary

## Axis 1 (Multi-hop QA)
HIRA achieved EM/F1 of {paper_numbers['hira_hotpotqa_em']:.2f}/{paper_numbers['hira_hotpotqa_f1']:.2f} on HotpotQA, with corresponding values for 2Wiki and MuSiQue saved in `paper_numbers.json`.

## Axis 2 (Multimodal Retrieval)
Image-node recall@5 is {paper_numbers['multimodal_recall_at_5']:.2f}% and multimodal gains are reflected in `eval/multimodal_summary.json`.

## Axis 3 (Incremental Update)
Maximum observed patch-vs-rebuild speedup factor is {paper_numbers['update_speedup_factor_max']:.2f}x from `eval/update_benchmark.csv`.

## Axis 4 (Agentic Stopping)
Average turns extracted from LangSmith traces: {paper_numbers['axis4_avg_turns']:.2f}.
"""
    (eval_dir / "results_summary.md").write_text(summary_md, encoding="utf-8")
    print("Generated table_main_results.tex, table_ablation.tex, eval/paper_numbers.json, eval/results_summary.md")


if __name__ == "__main__":
    main()

