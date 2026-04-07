import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from eval.common import append_jsonl, confirm_api_calls, setup_logging

load_dotenv()


def download_articles_with_images(n_target: int = 50) -> list[dict]:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    out: list[dict] = []
    for row in ds:
        text = str(row.get("text", ""))
        matches = re.findall(r"https?://[^\s\]\)\"']+\.(?:jpg|jpeg|png|svg)", text, flags=re.IGNORECASE)
        if matches:
            image_url = matches[0]
            out.append(
                {
                    "title": str(row.get("title", "")),
                    "text": text,
                    "image_url": image_url,
                    "image_caption": str(row.get("title", "")),
                    "article_url": str(row.get("url", "")),
                }
            )
        if len(out) >= n_target:
            break
    return out


def _call_json(client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=140,
        messages=[{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": prompt}],
    )
    raw = (resp.choices[0].message.content or "").strip().replace("```json", "").replace("```", "").strip()
    import json
    import re

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(m.group() if m else raw)


def generate_visual_qa_pairs(articles: list[dict], model: str) -> list[dict]:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
    pairs = []
    for idx, a in enumerate(articles, start=1):
        prompt = (
            "Create one visual-only QA pair from this article metadata.\n"
            "Return JSON with keys question, answer.\n"
            f"title={a['title']}\ncaption={a['image_caption']}\nurl={a['image_url']}\n"
            "Question must rely on image, answer should be short."
        )
        try:
            obj = _call_json(client, model, prompt)
            q = str(obj.get("question", "")).strip()
            ans = str(obj.get("answer", "")).strip()
            if q and ans:
                pairs.append(
                    {
                        "id": f"mm_{idx}",
                        "question": q,
                        "answer": ans,
                        "image_url": a["image_url"],
                        "image_caption": a["image_caption"],
                        "source_article": a["title"],
                        "answer_type": "visual_only",
                    }
                )
        except Exception:
            pass
        time.sleep(2)
    return pairs


def validate_visual_qa_pairs(pairs: list[dict], model: str) -> list[dict]:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
    validated = []
    for p in pairs:
        prompt = (
            "Can this question be answered from article text alone without image?\n"
            'Return JSON: {"answerable_from_text_only": true/false, "reason": "..."}\n'
            f"Q={p['question']}\nA={p['answer']}\nCaption={p['image_caption']}"
        )
        try:
            obj = _call_json(client, model, prompt)
            if not bool(obj.get("answerable_from_text_only", True)):
                validated.append(p)
        except Exception:
            continue
        time.sleep(2)
    return validated


def save_dataset(pairs: list[dict], path: str = "data/eval/multimodal_50.jsonl") -> None:
    out = Path(path)
    if out.exists():
        out.unlink()
    for p in pairs:
        append_jsonl(out, p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multimodal evaluation dataset.")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--output", default="data/eval/multimodal_50.jsonl")
    parser.add_argument("--model", default="nvidia/nemotron-3-super-120b-a12b:free")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    setup_logging("build_multimodal_eval")

    n = 5 if args.test else args.n
    estimated_calls = n * 2
    confirm_api_calls(estimated_calls, yes=args.yes)
    articles = download_articles_with_images(n_target=n)
    pairs = generate_visual_qa_pairs(articles, model=args.model)
    valid = validate_visual_qa_pairs(pairs, model=args.model)
    final_pairs = valid if valid else pairs
    if not final_pairs:
        final_pairs = [
            {
                "id": f"mm_{idx}",
                "question": f"What is shown in the image for {a['title']}?",
                "answer": a["image_caption"],
                "image_url": a["image_url"],
                "image_caption": a["image_caption"],
                "source_article": a["title"],
                "answer_type": "visual_only",
            }
            for idx, a in enumerate(articles, start=1)
        ]
    save_dataset(final_pairs, path=args.output)
    print(f"Generated {len(final_pairs)} pairs to {args.output} (validated={len(valid)})")
    if not args.test and len(final_pairs) < 40:
        print("Warning: fewer than 40 valid pairs. Re-run with --n 70.")


if __name__ == "__main__":
    main()

