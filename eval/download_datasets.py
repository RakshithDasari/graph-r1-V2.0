import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import load_dataset


SEED = 42
DEFAULT_SAMPLE_SIZE = 300


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: tuple[str, ...]
    config: str | None
    split: str
    out_path: Path
    dataset_tag: str


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="hotpotqa",
        aliases=("hotpot_qa",),
        config="distractor",
        split="validation",
        out_path=Path("data/eval/hotpotqa_300.jsonl"),
        dataset_tag="hotpotqa",
    ),
    DatasetSpec(
        name="2wikimhqa",
        aliases=("2WikiMultihopQA", "scholarly-shadows-syndicate/2wikimultihopqa"),
        config=None,
        split="validation",
        out_path=Path("data/eval/2wikimhqa_300.jsonl"),
        dataset_tag="2wikimhqa",
    ),
    DatasetSpec(
        name="musique",
        aliases=("MuSiQue", "dgslibisey/MuSiQue"),
        config=None,
        split="validation",
        out_path=Path("data/eval/musique_300.jsonl"),
        dataset_tag="musique",
    ),
)


def setup_logging() -> None:
    log_dir = Path("eval/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{pd.Timestamp.now():%Y-%m-%d_%H-%M}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def _pick_first(record: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def _normalize_supporting_facts(record: dict[str, Any]) -> list[str]:
    sf = _pick_first(record, ("supporting_facts", "supporting_facts_ids", "supportingfacts"), default=[])
    if isinstance(sf, dict):
        titles = sf.get("title") or sf.get("titles") or []
        if isinstance(titles, list):
            return [str(t).strip() for t in titles if str(t).strip()]
        return [str(titles).strip()] if str(titles).strip() else []
    if isinstance(sf, list):
        values: list[str] = []
        for item in sf:
            if isinstance(item, (list, tuple)) and item:
                values.append(str(item[0]).strip())
            elif isinstance(item, dict):
                val = _pick_first(item, ("title", "id", "passage_id"), "")
                if str(val).strip():
                    values.append(str(val).strip())
            elif str(item).strip():
                values.append(str(item).strip())
        return values
    return []


def _normalize_sample(record: dict[str, Any], dataset_tag: str) -> dict[str, Any]:
    sample_id = str(_pick_first(record, ("id", "_id", "qid", "question_id"), default="")).strip()
    question = str(_pick_first(record, ("question", "query"), default="")).strip()
    answer = _pick_first(record, ("answer", "answers", "gold_answer"), default="")
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    if isinstance(answer, dict):
        answer = _pick_first(answer, ("text", "answer"), default="")
    return {
        "id": sample_id,
        "question": question,
        "answer": str(answer).strip(),
        "supporting_facts": _normalize_supporting_facts(record),
        "dataset": dataset_tag,
    }


def _load_split(spec: DatasetSpec):
    last_err: Exception | None = None
    for alias in spec.aliases:
        try:
            ds = load_dataset(alias, spec.config, split=spec.split)
            logging.info("Loaded dataset %s via alias '%s' (%d rows)", spec.name, alias, len(ds))
            return ds
        except Exception as exc:
            last_err = exc
            logging.warning("Could not load %s via alias '%s': %s", spec.name, alias, exc)
    raise RuntimeError(f"Failed to load dataset {spec.name}. Last error: {last_err}")


def sample_and_normalize(spec: DatasetSpec, sample_size: int) -> tuple[list[dict[str, Any]], int]:
    dataset = _load_split(spec)
    total = len(dataset)
    if total < sample_size:
        raise ValueError(f"{spec.name} has only {total} rows, cannot sample {sample_size}.")
    rng = random.Random(SEED)
    indices = rng.sample(range(total), sample_size)
    normalized = [_normalize_sample(dataset[i], spec.dataset_tag) for i in indices]
    return normalized, total


def save_jsonl(samples: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and sample evaluation datasets.")
    parser.add_argument("--dry-run", action="store_true", help="Print first 3 normalized samples and do not save.")
    parser.add_argument("--test", action="store_true", help="Sample only 5 examples per dataset for a quick check.")
    args = parser.parse_args()
    setup_logging()

    sample_size = 5 if args.test else DEFAULT_SAMPLE_SIZE
    summary_rows: list[dict[str, Any]] = []

    for spec in DATASETS:
        samples, total = sample_and_normalize(spec, sample_size=sample_size)
        if args.dry_run:
            print(f"\n=== {spec.name} (first 3 normalized samples) ===")
            for row in samples[:3]:
                print(json.dumps(row, ensure_ascii=False))
        else:
            out_path = spec.out_path if not args.test else spec.out_path.with_name(spec.out_path.stem + "_test.jsonl")
            save_jsonl(samples, out_path)
            logging.info("Saved %d samples to %s", len(samples), out_path.as_posix())

        avg_q_len = sum(len(s["question"]) for s in samples) / len(samples)
        avg_a_len = sum(len(s["answer"]) for s in samples) / len(samples)
        summary_rows.append(
            {
                "dataset": spec.name,
                "total_available": total,
                "sampled_count": len(samples),
                "avg_question_len_chars": round(avg_q_len, 2),
                "avg_answer_len_chars": round(avg_a_len, 2),
            }
        )

    print("\n=== Summary ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()

