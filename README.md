# HIRA
### Hypergraph-Indexed Retrieval Augmentation via Multimodal Agentic Reasoning

<br>

![System Architecture](system_architecture.png)

<br>

## Overview

HIRA is a multimodal agentic retrieval-augmented generation system that extends the Graph-R1 architecture with three original contributions: native multimodal embedding support, a domain-agnostic LLM-based agentic controller, and an incremental hypergraph updater. The system constructs a knowledge hypergraph from documents and images, indexes it for dual-path vector search, and uses a multi-turn agentic loop to reason over retrieved context before generating answers.

---

## Contributions over Graph-R1

| Contribution | Graph-R1 | HIRA |
|---|---|---|
| Embedding | BGE text-only encoder | Gemini Embedding 2 — text, image, video, audio in a unified 3072-dim vector space |
| Agentic controller | GRPO-trained RL exit policy, domain-specific | Nvidia Nemotron via OpenRouter — LLM-based, domain-agnostic, zero retraining |
| Knowledge update | Static hypergraph, full rebuild on every update | Incremental diff-and-patch — cost proportional to new data only |

---

## Architecture

```
HIRA/
  graph/
    encoder.py      — Gemini Embedding 2 multimodal encoder
    builder.py      — 6-step knowledge base construction pipeline
    updater.py      — incremental diff-based hypergraph updater
  agent/
    retriever.py    — dual-path FAISS + one-hop NetworkX retrieval
    controller.py   — Nvidia Nemotron agentic reasoning loop
  main.py           — CLI entry point (build / query / update)
  app.py            — Streamlit web interface
```

---

## Embedding Model Options

HIRA supports two embedding backends depending on corpus type and cost constraints.

| Model | Dimensions | Modalities | Cost | Recommended for |
|---|---|---|---|---|
| Gemini Embedding 2 | 3072 | Text, image, video, audio | Free tier / pay-as-you-go | Mixed media corpora |
| Nvidia Nemotron Embed VL 1B v2 | 2048 | Text, document pages, tables, charts | Completely free | Text-only or document-heavy pipelines |

Switching between models requires changing two lines in `graph/encoder.py`. The rest of the pipeline is unchanged.

---

## Setup

**Requirements:** Python 3.11+

**API keys required:**
- Google AI Studio — [aistudio.google.com](https://aistudio.google.com)
- OpenRouter — [openrouter.ai](https://openrouter.ai)

```bash
git clone https://github.com/RakshithDasari/HIRA.git
cd HIRA
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
```

---

## Docker

> **TODO:** Add Docker setup and deployment instructions here.

---

## Usage

**Build a knowledge base from a document:**
```bash
python main.py --mode build --input data/sample/document.txt
```

**Query the knowledge base:**
```bash
python main.py --mode query --question "your question here"
```

**Update with new documents (incremental):**
```bash
python main.py --mode update --input data/sample/new_document.txt
```

**Launch the web interface:**
```bash
streamlit run app.py
```

---

## Performance

| Operation | Free tier | Paid tier |
|---|---|---|
| Build (500 words) | 2–3 min | ~10 sec |
| Query | 5–10 sec | 2–3 sec |
| Update (new document) | 1–2 min | ~5 sec |

Free-tier build time is governed by the Gemini API rate limit (10 RPM). This can be eliminated on a paid tier by setting `time.sleep(0)` in `graph/encoder.py`.

---

## Known Limitations

- Deletion is not supported — the updater is append-only. Production fix: migrate FAISS to `IndexIDMap` or replace with Milvus.
- Image entity descriptions are passed as file path text to the extraction LLM rather than actual image bytes. Full multimodal extraction requires a vision-capable extraction model.
- Free-tier build time scales linearly with document size due to per-request rate limiting.

---

## Dependencies

```
google-genai       Gemini Embedding 2
networkx           Hypergraph structure
numpy              Vector operations
faiss-cpu          Similarity search
python-dotenv      Environment variable loading
pillow             Image preprocessing
openai             OpenAI-compatible SDK for OpenRouter
pymupdf            PDF text and image extraction
streamlit          Web interface
pyvis              Interactive graph visualization
```

---

## References

- [Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning](https://arxiv.org/abs/2507.21892)
- [Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/embeddings)
- [Nvidia Nemotron Embed VL 1B v2](https://build.nvidia.com/nvidia/llama-nemotron-embed-vl-1b-v2)

---

## Author

**Rakshith Dasari**  
[GitHub](https://github.com/RakshithDasari) · [LinkedIn](https://linkedin.com/in/your-profile)

---

# Evaluation Journey 

This project was evaluated in a structured, step-by-step way so you can trust both the process and the outputs.

## What We Evaluated

We evaluated the system on 4 axes:

1. Multi-hop QA accuracy:
- Can the system answer complex questions that require connecting multiple facts?

2. Multimodal retrieval:
- Can it use image-related context when needed?

3. Incremental update efficiency:
- How fast does updating the graph/index get as new data is added?

4. Agentic stopping behavior:
- Does the controller stop at the right time, or over/under-run?

## Where Final Outputs Are

Final outputs are now organized into two folders:

1. `final_results/`
- Contains final row-level prediction outputs (JSONL files).

2. `final_summaries/`
- Contains final summaries, paper tables, benchmark/trace plots, and consolidated numbers.

## What Was Cleaned Up

- Test-only artifacts were removed.
- Optional rerun helpers were moved to `bin_optional/` and ignored via `.gitignore`.
- `requirements_eval.txt` was merged into `requirements.txt` so there is one requirements file.

## Important Note About Scores

Some runs were heavily affected by free-tier API rate limits (429 responses). That means the pipeline completed, but quality metrics may be lower than expected due to throttling rather than model design alone.

## If You Re-run Later

For best stability:

1. Run one evaluation job at a time.
2. Keep `--resume` enabled.
3. Re-generate final tables after runs finish.

This keeps the workflow reliable and reproducible.
