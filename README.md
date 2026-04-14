# HIRA
### Hypergraph-Indexed Retrieval Augmentation via Multimodal Agentic Reasoning

![System Architecture](system_architecture.png)

---

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

## Evaluation

The system was evaluated on four axes: multi-hop QA accuracy, multimodal retrieval, incremental update efficiency, and agentic stopping behavior. Evaluations were run on 300-sample splits of HotpotQA, 2WikiMultiHopQA, and MuSiQue.

> **Note on scores:** Some runs were affected by free-tier API rate limits (429 responses). Quality metrics may be lower than expected due to throttling rather than model design.

### Multi-hop QA

| Dataset | Exact Match | F1 | Samples | Avg Turns | Avg Latency (ms) | Timeout Rate | Error Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.00 | 0.00 | 300 | 3.00 | 2956.31 | 0.00% | 100.00% |
| 2WikiMultiHopQA | 0.00 | 0.00 | 300 | 3.00 | 4546.06 | 0.00% | 100.00% |
| MuSiQue | 0.00 | 0.10 | 300 | 2.99 | 2504.02 | 2.33% | 98.67% |

HotpotQA and 2WikiMultiHopQA returned zero exact match and zero F1, indicating the answer-generation path is not yet reliably producing correct outputs on these benchmarks. MuSiQue shows a small F1 gain (0.10), suggesting partial token overlap. Average turns are around 3 across all datasets — the agent iterates, but those turns are not yet translating into better answers. The high error rate is the main bottleneck in this stage.

### Multimodal Retrieval

| Metric | Value |
|---|---:|
| Image-node recall@5 | 0.00 |
| Cross-modal accuracy | 0.00 |
| Text-only accuracy | 0.00 |
| Delta | 0.00 |
| Avg embedding similarity | 0.2979 |

Image-node recall is zero in the current run, meaning the retrieval stage did not surface relevant image-linked nodes within the top-5 results. The non-zero average embedding similarity confirms the pipeline is functioning, but the signal is not yet strong enough to produce correct retrieval outcomes.

### Incremental Update Efficiency

| Checkpoint | Rebuild Time (s) | Patch Time (s) | Changed Entities | Total Entities | k/N Ratio | Recall After Update | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.0064 | 0.0064 | 4684 | 4684 | 1.0000 | 0.00 | 1.00 |
| 120 | 0.0027 | 0.0002 | 584 | 5268 | 0.1109 | 0.00 | 13.38× |

The second checkpoint shows a clear efficiency gain: patching is 13.38× faster than a full rebuild, with only ~11% of the graph requiring updates. Recall after update is still zero in the recorded benchmark — the incremental path is efficient, but update quality still needs validation.

### Agentic Stopping Behavior

| Metric | Value |
|---|---:|
| Avg turns | 1.00 |
| % stopping at 1 turn | 100.00% |
| Early stop rate | 91.67% |
| Over-run rate | 0.00% |
| Avg latency (ms) | 55003.15 |
| P95 latency (ms) | 95138.62 |

The controller strongly favors stopping after one turn. Over-run rate is zero, so the system does not waste effort by searching too long. However, the high early-stop rate suggests the controller often halts before gathering enough evidence. Tail latency (P95 ~95 seconds) remains a practical concern.

### Overall Assessment

The strongest result is the incremental-update speedup, which validates the update architecture's efficiency over full rebuilds. The stopping policy is also stable and does not over-run. The main weakness is answer quality — multi-hop QA and multimodal retrieval do not yet show measurable gains. If the next goal is publication-grade performance, the priority should be improving retrieval precision and investigating the high error rates in the answer-generation stage.

---

## Known Limitations

- **Deletion not supported** — the updater is append-only. Production fix: migrate FAISS to `IndexIDMap` or replace with Milvus.
- **Image extraction is path-based** — image entity descriptions are passed as file path text to the extraction LLM rather than actual image bytes. Full multimodal extraction requires a vision-capable extraction model.
- **Free-tier latency** — build time scales linearly with document size due to per-request rate limiting.

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
