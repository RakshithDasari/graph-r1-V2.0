import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from dotenv import load_dotenv
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


load_dotenv()
from openai import OpenAI

from eval.common import append_jsonl, confirm_api_calls, load_jsonl, setup_logging


SYSTEM_PROMPT = (
    "You are a QA evaluator. Given a question, a predicted answer, and the correct answer, "
    'decide if the prediction is correct. Respond ONLY with valid JSON: {"correct": true/false, "confidence": 0.0-1.0}. '
    "Be lenient with paraphrasing but strict with factual content."
)


class LLMJudge:
    def __init__(self, model: str = "nvidia/nemotron-3-super-120b-a12b:free"):
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=60.0,
            max_retries=2,
        )

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = match.group() if match else raw
        parsed = json.loads(payload)
        return {
            "correct": bool(parsed.get("correct", False)),
            "confidence": float(parsed.get("confidence", 0.0)),
        }

    def judge_single(self, question: str, prediction: str, ground_truth: str) -> dict[str, Any]:
        user_prompt = (
            f"Question: {question[:80]}\n"
            f"Prediction: {prediction[:80]}\n"
            f"Correct answer: {ground_truth[:80]}\n"
            "Return JSON only."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=60,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = (getattr(response.choices[0].message, "content", "") or "").strip()
        try:
            parsed = self._extract_json(raw)
            return {"correct": parsed["correct"], "confidence": parsed["confidence"], "reasoning": ""}
        except Exception:
            return {"correct": False, "confidence": 0.0, "reasoning": "parse_error"}

    def judge_batch(
        self,
        samples: list[dict],
        batch_size: int = 20,
        output_path: str | Path | None = None,
        partial_path: str | Path = "eval/judge_partial.jsonl",
    ) -> list[dict[str, Any]]:
        judged: list[dict] = []
        output_path = Path(output_path) if output_path is not None else None
        partial_path = Path(partial_path)
        partial_path.parent.mkdir(parents=True, exist_ok=True)

        if partial_path.exists():
            partial_path.unlink()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not output_path.exists():
                output_path.write_text("", encoding="utf-8")

        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            for sample in batch:
                result = self.judge_single(
                    question=sample.get("question", ""),
                    prediction=sample.get("prediction", ""),
                    ground_truth=sample.get("ground_truth", ""),
                )
                merged = dict(sample)
                merged["llm_correct"] = result["correct"]
                merged["llm_confidence"] = result["confidence"]
                judged.append(merged)
                if output_path is not None:
                    append_jsonl(output_path, merged)
                if len(judged) % 50 == 0:
                    with partial_path.open("w", encoding="utf-8") as f:
                        for row in judged:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                time.sleep(0.5)

        with partial_path.open("w", encoding="utf-8") as f:
            for row in judged:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return judged


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge for semantic QA correctness.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="nvidia/nemotron-3-super-120b-a12b:free")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true", help="Only judge 5 samples.")
    parser.add_argument("--yes", action="store_true", help="Auto-confirm API-call prompt.")
    args = parser.parse_args()

    log = setup_logging("llm_judge")
    in_rows = load_jsonl(args.input)
    if args.test:
        in_rows = in_rows[:5]
    confirm_api_calls(len(in_rows), yes=args.yes)

    already: set[str] = set()
    output_rows: list[dict[str, Any]] = []
    if args.resume and os.path.exists(args.output):
        output_rows = load_jsonl(args.output)
        already = {str(r.get("id", "")) for r in output_rows}
    elif args.resume and Path("eval/judge_partial.jsonl").exists():
        output_rows = load_jsonl("eval/judge_partial.jsonl")
        already = {str(r.get("id", "")) for r in output_rows}

    pending = [r for r in in_rows if str(r.get("id", "")) not in already]
    judge = LLMJudge(model=args.model)
    if output_rows and args.resume:
        with open(args.output, "w", encoding="utf-8") as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    judged = judge.judge_batch(pending, batch_size=args.batch_size, output_path=args.output)

    total = len(output_rows) + len(judged)
    em0_saved = 0
    for row in output_rows + judged:
        if int(row.get("em", row.get("exact_match", 0))) == 0 and bool(row.get("llm_correct", False)):
            em0_saved += 1
    final_acc = (sum(1 for r in output_rows + judged if bool(r.get("llm_correct", False))) / total * 100.0) if total else 0.0
    est_input_tokens = total * 120
    est_cost = (est_input_tokens / 1000.0) * 0.00025
    print(f"Total judged: {total}")
    print(f"Saved (EM=0 but LLM correct): {(em0_saved/total*100.0) if total else 0.0:.2f}%")
    print(f"Final accuracy with LLM correction: {final_acc:.2f}%")
    print(f"Estimated cost at $0.00025/1k input tokens: ${est_cost:.6f}")
    log.info("Wrote judged outputs to %s", args.output)


if __name__ == "__main__":
    main()

