# HIRA Evaluation Prompt Pack
# Hand this file to any capable coding agent (GPT-4o, o3, Gemini 2.5 Pro, Claude)
# Each prompt is self-contained. Run them in order. One prompt = one session.
# Free-tier safe: designed around HuggingFace free datasets, NVIDIA free Nemotron,
# Gemini free embedding quota (10 RPM), and Claude Haiku as LLM judge.

---

## CONTEXT (paste this at the top of EVERY session)

```
You are helping evaluate HIRA — a Hypergraph-Indexed Retrieval Augmentation system.
HIRA uses:
- Gemini Embedding 2 (gemini-embedding-2-preview) for multimodal embeddings (10 RPM free quota)
- NVIDIA Nemotron Super 120B via OpenRouter for entity extraction + agentic controller
- FAISS (faiss-cpu) for vector search
- NetworkX for hypergraph structure
- LangSmith for observability (already instrumented)

The goal is to produce paper-quality evaluation results across 4 axes:
  Axis 1: Multi-hop QA accuracy (EM + F1) on standard datasets
  Axis 2: Multimodal retrieval (image-node recall@5, cross-modal accuracy)
  Axis 3: Incremental update efficiency (latency, k/N ratio, recall preservation)
  Axis 4: Agentic stopping quality (avg turns, early-stop rate, over-run rate)

All code must be production-quality: type hints, docstrings, error handling, logging.
Respect all API rate limits. Never exceed 10 RPM for Gemini embedding calls.
Save all results to JSON and CSV for reproducibility.
```

---

## PROMPT 1 — Dataset Downloader + Sampler
## Goal: Download HotpotQA, 2WikiMultiHopQA, MuSiQue. Sample 300 Qs each. Zero cost.

```
[paste CONTEXT block above first]

Task: Write a Python script `eval/download_datasets.py` that does the following:

1. Uses HuggingFace `datasets` library to download:
   - HotpotQA: split="validation", config="distractor"
   - 2WikiMultiHopQA: split="validation"
   - MuSiQue: split="validation"

2. From each dataset, samples exactly 300 questions using a fixed random seed (seed=42)
   to ensure reproducibility across runs.

3. For each sample, extracts and normalises these fields into a standard schema:
   {
     "id": str,
     "question": str,
     "answer": str,          # ground truth answer string
     "supporting_facts": list[str],  # list of passage titles or IDs if available
     "dataset": str          # "hotpotqa" | "2wikimhqa" | "musique"
   }

4. Saves three files:
   - data/eval/hotpotqa_300.jsonl
   - data/eval/2wikimhqa_300.jsonl
   - data/eval/musique_300.jsonl

5. Prints a summary table: dataset name, total available, sampled count,
   avg question length (chars), avg answer length (chars).

Requirements:
- Handle missing fields gracefully (some datasets have different schema keys)
- Add a --dry-run flag that prints the first 3 samples from each dataset
  without saving anything
- All file paths relative to project root
- Dependencies: datasets, pandas. No other installs needed.
- Include a requirements_eval.txt with pinned versions.

Do not use any paid API. This is pure local data processing.
```

---

## PROMPT 2 — EM and F1 Scorer
## Goal: Standard NLP evaluation metrics, identical to SQuAD/HotpotQA official scorers.

```
[paste CONTEXT block above first]

Task: Write `eval/metrics.py` — a self-contained metrics module with no external
NLP library dependencies (no nltk, no spacy, no transformers).

Implement these functions exactly:

1. normalize_answer(answer: str) -> str
   - Lowercase
   - Remove punctuation (string.punctuation)
   - Remove articles: "a", "an", "the"
   - Collapse whitespace
   - This must exactly match the official HotpotQA evaluation script logic.

2. exact_match(prediction: str, ground_truth: str) -> int
   - Returns 1 if normalize_answer(prediction) == normalize_answer(ground_truth)
   - Returns 0 otherwise

3. f1_score(prediction: str, ground_truth: str) -> float
   - Token-level F1 between normalized prediction and normalized ground truth
   - precision = common_tokens / len(prediction_tokens)
   - recall = common_tokens / len(ground_truth_tokens)
   - F1 = 2 * precision * recall / (precision + recall) if denom > 0 else 0.0

4. score_dataset_results(results: list[dict]) -> dict
   - Input: list of {"prediction": str, "ground_truth": str, "id": str}
   - Output: {
       "exact_match": float,   # percentage 0-100
       "f1": float,            # percentage 0-100
       "n": int,
       "em_per_sample": list[int],
       "f1_per_sample": list[float]
     }

5. compute_confidence_interval(scores: list[float], confidence: float = 0.95) -> tuple[float, float]
   - Bootstrap confidence interval (n_bootstrap=1000, seed=42)
   - Returns (lower_bound, upper_bound)

Write a __main__ block that runs a self-test with 5 hard-coded example pairs
and prints expected vs actual scores. All 5 must pass.

No external imports beyond standard library + numpy.
```

---

## PROMPT 3 — LLM-as-Judge (Claude Haiku)
## Goal: Cheap, fast correctness judgment for open-ended answers where EM/F1 fails.

```
[paste CONTEXT block above first]

Task: Write `eval/llm_judge.py` using the Anthropic Python SDK.

The script judges whether HIRA's answer is semantically correct given the
ground truth, for cases where exact match fails but the answer is still right
(e.g. "Barack Obama" vs "Obama").

Implement:

1. class LLMJudge:
   def __init__(self, model: str = "claude-haiku-4-5-20251001"):
       # Use claude-haiku — cheapest, fast enough for 900 judgments
   
   def judge_single(self, question: str, prediction: str, ground_truth: str) -> dict:
       # Returns {"correct": bool, "confidence": float, "reasoning": str}
       # Prompt must be extremely tight — max 200 tokens input, 60 tokens output
       # Ask model to respond in JSON only: {"correct": true/false, "confidence": 0.0-1.0}
   
   def judge_batch(self, samples: list[dict], batch_size: int = 20) -> list[dict]:
       # Each sample: {"id", "question", "prediction", "ground_truth"}
       # Adds "llm_correct" and "llm_confidence" fields to each sample
       # Adds 0.5s sleep between calls to avoid rate limits
       # Saves partial results to eval/judge_partial.jsonl every 50 samples
       #   so you can resume if interrupted
       # Returns full list with judgments added

2. A CLI: python eval/llm_judge.py --input eval/results_hotpotqa.jsonl --output eval/judged_hotpotqa.jsonl

3. At the end, prints:
   - Total judged
   - % where EM=0 but LLM judged correct (the "saved" cases)
   - Final accuracy with LLM correction applied
   - Estimated cost at $0.00025/1k input tokens

The system prompt for the judge must be:
"You are a QA evaluator. Given a question, a predicted answer, and the correct answer,
decide if the prediction is correct. Respond ONLY with valid JSON: {"correct": true/false, "confidence": 0.0-1.0}.
Be lenient with paraphrasing but strict with factual content."

Do not let the judge see the reasoning chain — predictions and ground truths only.
```

---

## PROMPT 4 — HIRA Batch Runner for Axis 1
## Goal: Run HIRA on all 900 sampled questions. Respect rate limits. Save everything.

```
[paste CONTEXT block above first]

Task: Write `eval/run_hira_eval.py` — the main evaluation harness for Axis 1.

This script runs HIRA's full pipeline (build → query) on each of the 900 sampled
questions and records results. HIRA's existing modules are at:
  graph/encoder.py       (GeminiEncoder)
  graph/builder.py       (HypergraphBuilder)
  agent/retriever.py     (Retriever)
  agent/controller.py    (AgenticController)

Implement:

1. class HIRAEvaluator:
   def __init__(self, index_dir: str, langsmith_project: str):
       # Load pre-built HIRA index from index_dir
       # Initialise LangSmith client for trace logging
   
   def query_with_timeout(self, question: str, timeout_seconds: int = 30) -> dict:
       # Wraps AgenticController.run() with a timeout
       # Returns {"answer": str, "turns": int, "latency_ms": float, "error": str|None}
   
   def run_dataset(self, dataset_path: str, output_path: str, resume: bool = True) -> None:
       # Loads JSONL, skips already-processed IDs if resume=True
       # For each question: calls query_with_timeout
       # Writes result immediately to output JSONL (one line per question)
       # Sleeps 6 seconds between calls (10 RPM Gemini quota = 1 call/6s for safety)
       # Prints live progress: "42/300 | EM so far: 34.2% | avg latency: 4.2s"
       # On KeyboardInterrupt: saves progress and exits cleanly

2. CLI:
   python eval/run_hira_eval.py \
     --index-dir data/hira_index \
     --dataset data/eval/hotpotqa_300.jsonl \
     --output eval/results_hotpotqa.jsonl \
     --resume

3. After all questions done, compute and print:
   - EM, F1 using eval/metrics.py
   - Avg turns per query
   - Avg latency per query (ms)
   - Timeout rate (%)
   - Error rate (%)

4. Save a summary JSON to eval/summary_hotpotqa.json

CRITICAL rate limit rules:
- Never make more than 9 Gemini embedding calls per minute (stay under 10 RPM limit)
- Track embedding call timestamps in a rolling window, sleep if needed
- Log every rate limit sleep to console with duration

The script must be safe to interrupt and resume at any point.
```

---

## PROMPT 5 — Baseline Runners
## Goal: Run GraphRAG-style flat RAG and fixed-policy baselines for comparison.

```
[paste CONTEXT block above first]

Task: Write `eval/baselines.py` with two simple baseline retrievers to compare against HIRA.

Baseline 1 — FlatRAG (chunk-based, no graph):
- Takes the same source documents HIRA was built from
- Chunks them with 500-word window, 100-word overlap (same as HIRA)
- Embeds chunks with Gemini Embedding 2 (same encoder as HIRA for fair comparison)
- At query time: embed query, retrieve top-5 chunks by cosine similarity
- Passes retrieved chunks as context to Nemotron via OpenRouter for answer generation
- Single-turn only (no agentic loop)

Baseline 2 — FixedTurnHIRA (HIRA with forced 1-turn stopping):
- Uses HIRA's full hypergraph index (same as main eval)
- BUT: always stops after exactly 1 retrieval turn regardless of AgenticController judgment
- This isolates the value of the agentic multi-turn loop

class FlatRAGBaseline:
    def __init__(self, docs_dir: str, index_path: str):
        ...
    def build(self) -> None: ...  # embed + index chunks
    def query(self, question: str) -> dict: ...  # returns {"answer", "latency_ms"}

class FixedTurnHIRA:
    def __init__(self, index_dir: str):
        ...
    def query(self, question: str, n_turns: int = 1) -> dict: ...

Both must:
- Respect the 10 RPM Gemini quota with the same rate limiting logic as run_hira_eval.py
- Return the exact same output schema as HIRAEvaluator.query_with_timeout()
- Be runnable via CLI:
  python eval/baselines.py --baseline flat_rag --dataset data/eval/hotpotqa_300.jsonl --output eval/results_flatrag_hotpotqa.jsonl
  python eval/baselines.py --baseline fixed_turn --dataset data/eval/hotpotqa_300.jsonl --output eval/results_fixed_hotpotqa.jsonl
```

---

## PROMPT 6 — Axis 3: Incremental Update Benchmark
## Goal: Measure O(k) vs O(N) update cost. Pure local, zero API cost after index built.

```
[paste CONTEXT block above first]

Task: Write `eval/benchmark_update.py` — the Axis 3 evaluation. This is entirely
local computation after the initial index is built. No API calls during timing.

The experiment:
- Start with a base corpus of 100 Wikipedia articles (download from HuggingFace
  `wikimedia/wikipedia` dataset, 20231101.en split, first 100 articles)
- Build HIRA index on these 100 articles (one-time cost, not measured)
- Then run incremental updates: add 50 more articles at a time up to 500 total
- At each checkpoint (100, 150, 200, ..., 500 articles):
  * Measure wall-clock time for IncrementalUpdater.update() (the patch time)
  * Measure wall-clock time for a full HypergraphBuilder rebuild from scratch
  * Record k (new entities added) and N (total corpus size)
  * Run 20 test queries and measure entity recall@5 to verify no regression

Implement:

1. download_wiki_articles(n: int, seed: int = 42) -> list[dict]
   # Returns list of {"title": str, "text": str} from HuggingFace wikipedia dataset

2. time_full_rebuild(articles: list[dict]) -> dict
   # Returns {"rebuild_time_s": float, "n_entities": int, "n_hyperedges": int}
   # IMPORTANT: disable all Gemini API calls during timing — use pre-computed embeddings
   # stored in a cache dict to isolate the pure rebuild overhead

3. time_incremental_update(new_articles: list[dict], existing_index_dir: str) -> dict
   # Returns {"update_time_s": float, "k_new_entities": int, "k_new_hyperedges": int}
   # Same: use cached embeddings so only the diff+patch logic is timed

4. measure_recall_after_update(index_dir: str, test_queries: list[dict]) -> float
   # Returns entity recall@5 averaged over test_queries

5. run_benchmark() -> None
   # Runs the full experiment, saves results to eval/update_benchmark.csv
   # Columns: checkpoint_n, rebuild_time_s, patch_time_s, k_entities, N_entities,
   #           k_over_N_ratio, recall_after_update, speedup_factor

6. plot_results(csv_path: str) -> None
   # Generates eval/update_benchmark.png using matplotlib only
   # Two subplots: (1) patch time vs rebuild time as corpus grows
   #               (2) k/N ratio as corpus grows
   # Paper-quality: 300 DPI, serif font, proper axis labels, legend

This benchmark must be completely reproducible with seed=42 everywhere.
No randomness in timing measurements — run each 3 times and report mean.
```

---

## PROMPT 7 — Axis 2: Multimodal Evaluation Dataset Builder
## Goal: Build the 50-sample multimodal eval set from scratch since none exists.

```
[paste CONTEXT block above first]

Task: Write `eval/build_multimodal_eval.py` to create a small but rigorous
multimodal evaluation dataset for Axis 2.

The dataset needs 50 Q&A pairs where the correct answer REQUIRES reading an
embedded image (chart, diagram, or table-as-image). This is a novel dataset
you are creating — no prior benchmark exists for multimodal graph RAG.

Strategy — use Wikipedia articles that contain figures with captions:

1. download_articles_with_images(n_target: int = 50) -> list[dict]
   # Uses HuggingFace `wikimedia/wikipedia` 
   # Filters for articles that have images with informative captions
   # Downloads the image URLs (do NOT download the actual images here — just URLs)
   # Returns list of {"title", "text", "image_url", "image_caption", "article_url"}

2. generate_visual_qa_pairs(articles: list[dict]) -> list[dict]
   # For each article, uses Nemotron via OpenRouter to generate 1 Q&A pair where:
   #   - The question can ONLY be answered by looking at the image
   #   - The answer is a short factual string (not a paragraph)
   #   - The question is NOT answerable from the article text alone
   # Returns list of {"id", "question", "answer", "image_url", "image_caption",
   #                   "source_article", "answer_type": "visual_only"}
   # Filter out any pairs where the answer appears verbatim in the article text

3. validate_visual_qa_pairs(pairs: list[dict]) -> list[dict]
   # Double-checks each pair: asks Nemotron "Can this question be answered from
   # text alone without the image?" — keeps only pairs where answer is "No"
   # Logs rejected pairs with reason

4. save_dataset(pairs: list[dict], path: str = "data/eval/multimodal_50.jsonl")

5. CLI: python eval/build_multimodal_eval.py --n 50 --output data/eval/multimodal_50.jsonl

Rate limit note: this script makes ~100 Nemotron calls total. OpenRouter free tier
allows sufficient calls. Add 2s sleep between calls. Expected runtime: ~15 minutes.

The final dataset must have at least 40 valid pairs after validation filtering.
If fewer than 40, increase --n to 70 and re-run.
```

---

## PROMPT 8 — Axis 2: Multimodal Retrieval Evaluator

```
[paste CONTEXT block above first]

Task: Write `eval/eval_multimodal.py` to evaluate HIRA on the multimodal dataset.

For each of the 50 multimodal Q&A pairs:

1. Index the image into HIRA:
   - Download the image from image_url and save temporarily
   - Call GeminiEncoder.encode(image_path) to embed it
   - Add it as a node to the hypergraph with metadata {"type": "image", "caption": ...}
   - This is the "with image" condition

2. For the "without image" condition (ablation baseline):
   - Use the exact same HIRA index but WITHOUT the image node
   - Only text nodes from the article are present

3. For each query, measure:
   - image_node_in_top5: bool — did the image node appear in FAISS top-5 results?
   - answer_correct_with_image: bool — LLM judge correctness when image is available
   - answer_correct_without_image: bool — LLM judge correctness without image
   - embedding_similarity: float — cosine similarity of query to image node embedding

4. Aggregate and report:
   - image_node_recall_at_5: float (% of queries where image node in top 5)
   - cross_modal_accuracy: float (% correct when image available)
   - text_only_accuracy: float (% correct without image)
   - delta: cross_modal_accuracy - text_only_accuracy (the multimodal gain)
   - avg_embedding_similarity: float

5. Save full results to eval/multimodal_results.json
   Save summary to eval/multimodal_summary.json

This directly fills in Table II of the paper.
Use LLMJudge from eval/llm_judge.py for correctness assessment.
Respect 10 RPM Gemini quota — each query makes 2 embedding calls (query + image).
```

---

## PROMPT 9 — LangSmith Trace Extractor for Axis 4
## Goal: Extract agentic stopping metrics from your existing LangSmith traces.

```
[paste CONTEXT block above first]

Task: Write `eval/extract_langsmith_metrics.py` to pull evaluation data from
the LangSmith traces you've already collected.

The script connects to LangSmith API and extracts per-run metrics.

1. class LangSmithExtractor:
   def __init__(self, project_name: str, api_key: str):
       from langsmith import Client
       self.client = Client(api_key=api_key)
       self.project = project_name
   
   def get_runs(self, run_type: str = "chain", limit: int = 500) -> list[dict]:
       # Fetches runs from LangSmith project
       # Filters to HIRA AgenticController runs only
       # Returns list of raw run objects
   
   def extract_turn_count(self, run: dict) -> int:
       # Parses the run's child spans to count retrieval turns
       # A "turn" = one call to Retriever.search() within the run
   
   def extract_metrics_from_runs(self, runs: list[dict]) -> pd.DataFrame:
       # For each run, extracts:
       # - run_id, question, final_answer
       # - total_latency_ms
       # - n_turns (from extract_turn_count)
       # - was_early_stop: bool (n_turns < max_turns AND done=True)
       # - was_over_run: bool (n_turns == max_turns AND answer quality low)
       # - embedding_calls: int (count of GeminiEncoder calls)
       # - retrieval_hit_rate: float (entities_returned / entities_searched)
       # Returns DataFrame

2. def compute_stopping_stats(df: pd.DataFrame) -> dict:
   # Returns:
   # - avg_turns: float
   # - pct_1_turn: float  (resolved in single turn)
   # - pct_2_turns: float
   # - pct_3_turns: float (hit max)
   # - early_stop_rate: float (done before max_turns)
   # - over_run_rate: float (hit max without confident answer)
   # - avg_latency_ms: float
   # - p95_latency_ms: float

3. CLI:
   python eval/extract_langsmith_metrics.py \
     --project hira-eval \
     --output eval/agentic_stopping_stats.json

4. Prints a formatted summary table ready to paste into the paper.

Also generate eval/turn_distribution.png — a bar chart showing
% of queries resolved at turn 1, 2, 3 respectively.
Uses matplotlib only. 300 DPI, paper-quality styling.
```

---

## PROMPT 10 — Results Aggregator + Paper Table Generator
## Goal: Combine all eval results into LaTeX-ready Table II.

```
[paste CONTEXT block above first]

Task: Write `eval/generate_paper_tables.py` — the final step that reads all
result files and generates LaTeX-formatted tables for the paper.

Input files (all produced by previous eval scripts):
- eval/summary_hotpotqa.json
- eval/summary_2wikimhqa.json
- eval/summary_musique.json
- eval/judged_hotpotqa.jsonl (with LLM corrections)
- eval/judged_2wikimhqa.jsonl
- eval/judged_musique.jsonl
- eval/results_flatrag_hotpotqa.jsonl
- eval/results_fixed_hotpotqa.jsonl
- eval/multimodal_summary.json
- eval/update_benchmark.csv
- eval/agentic_stopping_stats.json

Outputs:

1. table_main_results.tex
   The main comparison table (replaces Table II in the paper):
   - Rows: HIRA, FlatRAG baseline, FixedTurn-HIRA baseline
   - Columns per dataset: EM, F1, Avg-Turns, Latency(s)
   - Bottom rows: Multimodal recall@5, Update speedup factor
   - Format: proper \hline, \textbf for best result per column
   - Caption already written

2. table_ablation.tex
   Ablation table:
   - HIRA full vs HIRA no-multimodal vs HIRA fixed-turn
   - Shows contribution of each component

3. eval/paper_numbers.json
   All key numbers in one flat JSON for easy reference when writing.
   E.g. {"hira_hotpotqa_em": 42.3, "hira_hotpotqa_f1": 51.7, ...}

4. eval/results_summary.md
   Human-readable markdown summary of all results with interpretation.
   One paragraph per axis.

The LaTeX must compile cleanly in a standard IEEE conference template.
Use \multicolumn and \multirow where appropriate.
Boldface the best result in every column.
Include confidence intervals as ± values where computed.
```

---

## PROMPT 11 — Rate Limit Guardian (utility, use in all scripts)
## Goal: Reusable rate limiter to protect the 10 RPM Gemini quota across all scripts.

```
[paste CONTEXT block above first]

Task: Write `eval/rate_limiter.py` — a thread-safe rate limiter to be imported
by all other eval scripts.

class RateLimiter:
    def __init__(self, max_calls_per_minute: int = 9):
        # Use 9 not 10 — stay safely under the limit
        # Uses a deque of call timestamps with maxlen = max_calls_per_minute
    
    def wait_if_needed(self) -> float:
        # Checks if adding another call would exceed the rate
        # If yes: sleeps for exactly the right duration, logs wait time
        # If no: records timestamp and returns immediately
        # Returns: how many seconds it slept (0.0 if no wait needed)
    
    def __call__(self, func):
        # Decorator: wraps any function with rate limiting
        # Usage: @rate_limiter on GeminiEncoder.encode()

Also implement:
- global_gemini_limiter = RateLimiter(max_calls_per_minute=9)
  (a singleton to be imported by all scripts)

- A context manager version:
  with rate_limiter:
      result = encoder.encode(text)

Unit test: in __main__, simulate 15 calls and verify total time >= 60s
and no more than 9 calls were made in any 60-second window.
Print a timeline showing each call and any sleeps.
```

---

## EXECUTION ORDER

Run these in sequence. Each builds on the previous.

1.  `download_datasets.py`        — ~5 min, no API calls
2.  `rate_limiter.py`             — instant, no API calls  
3.  `metrics.py`                  — instant, no API calls
4.  `llm_judge.py`                — test with 5 samples first
5.  `benchmark_update.py`         — ~30 min, minimal API calls (cached embeddings)
6.  `baselines.py` (build index)  — ~2 hrs at 10 RPM, start overnight
7.  `run_hira_eval.py` (hotpotqa) — ~30 min at 10 RPM
8.  `run_hira_eval.py` (2wiki)    — ~30 min at 10 RPM
9.  `run_hira_eval.py` (musique)  — ~30 min at 10 RPM
10. `build_multimodal_eval.py`    — ~15 min
11. `eval_multimodal.py`          — ~20 min at 10 RPM
12. `extract_langsmith_metrics.py`— instant (reads existing traces)
13. `generate_paper_tables.py`    — instant

Total API cost estimate at free tier:
- Gemini embeddings: free (quota-limited, not cost-limited)
- Nemotron via OpenRouter: ~$0.80 total for 900 queries + baselines
- Claude Haiku judge: ~$0.20 total for 900 judgments
- Grand total: under $1.50 USD

---

## IMPORTANT NOTES FOR THE AGENT

1. Every script must have a --dry-run or --test flag that runs on 5 samples only.
   ALWAYS test with --test before running the full eval. This protects your quota.

2. Every script must save progress incrementally and support --resume.
   A crash at query 280/300 should not require restarting from zero.

3. Rate limit the Gemini encoder at 9 RPM maximum. Import RateLimiter from
   eval/rate_limiter.py in every script that calls GeminiEncoder.

4. Never hardcode API keys. Read from environment variables:
   GOOGLE_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY, LANGSMITH_API_KEY

5. Log everything to eval/logs/YYYY-MM-DD_HH-MM.log using Python logging module.
   Console output should be clean progress bars / summaries only.

6. All random operations use seed=42. Results must be identical on re-run.

7. Before running any new script, print the estimated number of API calls
   it will make and ask for confirmation.