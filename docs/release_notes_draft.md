# Adaptive Agentic RAG v1.0.0 — Release Notes Draft

> **Canonical V2-A Production Release**: Multi-Hop Agentic RAG with Hybrid Retrieval, Cross-Encoder Reranking, Evidence-Aware Recovery, Grounded Generation, Semantic Safety Verification, and Production FastAPI Service.

---

## Release Highlights

- **Canonical V2-A Architecture**: Full frozen multi-stage pipeline combining dense semantic vectors (Qwen-0.6B) and sparse lexical signals (BM25Okapi) via Reciprocal Rank Fusion ($k=60$), cross-attention reranking (BGE-reranker-base), and Maximal Marginal Relevance diversity filtering ($\lambda=0.7$, top 5).
- **Evidence Gating & Adaptive Recovery**: Enforces strict entity anchor and publisher coverage guards prior to LLM synthesis; triggers targeted query rewriting and semantic rescue only when structural misses are recoverable.
- **NLI Claim Grounding & Fail-Closed Semantic Verification**: Deconstructs draft answers into atomic propositions, verifies premise entailment via DeBERTa-v3 NLI ($P \ge 0.70$), and conservatively suppresses unsupported or asymmetric direct answers to `UNKNOWN`.
- **Production FastAPI Service**: Delivers non-blocking asynchronous inference, concurrency serialization, verified citation binding, and structured observability traces.
- **Interactive Visual Showcase**: Pure HTML/CSS/JS dark-themed engineering dashboard (`demo/`) visualizing the 7-node LangGraph execution flow with live latency telemetry and citation inspection.
- **Automated CI Workflow**: GitHub Actions pipeline running 121 portable unit and integration tests under Python 3.12 without external GPU or database dependencies.
- **100% Test Regression**: Complete suite of 179 tests passing with 0 failures.

---

## Architecture Specification

The system implements the **V2-A (Frozen Canonical)** architecture:

```text
Query
  │
  ├──► Query Router (Simple vs. Complex Classification)
  │
  ├──► Hybrid Retrieval (Dense Qwen3-0.6B + Sparse BM25Okapi, RRF k=60)
  │
  ├──► Cross-Encoder Reranking (BAAI/bge-reranker-base + MMR λ=0.7, top-5)
  │
  ├──► Evidence Gating & Adaptive Recovery
  │      ├── Explicit Source Coverage Guard
  │      ├── Corpus Source Availability Check
  │      └── Source-Targeted Rewrite & Semantic Rescue (Max 1 attempt)
  │
  ├──► Single-Pass Structured Generation (Qwen2.5-1.5B-Instruct ChatML)
  │
  ├──► Propositional Claim Grounding (DeBERTa-v3-small NLI, P ≥ 0.70)
  │
  ├──► Semantic Safety Verification (StructuredConclusionVerifier)
  │      ├── Supported: Verified Direct Answer + Grounded Citations
  │      └── Unsupported: Fail-Closed Safe Abstention (UNKNOWN)
```

---

## Key Performance & Evaluation Metrics

Evaluated on the isolated `final_untouched_test.json` benchmark (100 multi-hop questions):

| Metric | Measured Value | Scope / Definition |
| :--- | :---: | :--- |
| **Recall@10** | **0.8870** | Gold context retrieval across multi-hop queries |
| **nDCG@10** | **0.8421** | Ranking quality of top-10 candidate chunks |
| **Null / Unanswerable Abstention** | **100.0%** | Zero hallucinations on out-of-corpus queries |
| **Citation Validity** | **100.0%** | All asserted claims backed by entailed source passages ($P \ge 0.70$) |
| **Answer Coverage** | **68.0%** | Percentage of queries where evidence was sufficient to answer |
| **Answered Accuracy** | **83.8%** | Direct answer precision on answered subset |
| **Overall Dataset Accuracy** | **57.0%** | Strict global accuracy under fail-closed safety gating |

---

## FastAPI Production Service

- **GET `/health`**: Liveness probe returning HTTP 200 `{"status": "ok"}`.
- **GET `/ready`**: Readiness probe confirming pre-warmed transformer backbones and Qdrant collection `multihop_chunks_v2`.
- **GET `/v1/system`**: System architecture metadata, corpus chunk count (8,173), and device diagnostics.
- **POST `/v1/query`**: Core inference endpoint accepting `{ "query": str, "include_trace": bool }` and returning direct propositions, formatted text, grounded citations, latency profiles, and stage trace telemetry.

---

## CI & Automated Testing

- **Portable CI Suite**: 121 unit & integration tests running on GitHub Actions (`ubuntu-latest`, Python 3.12, `uv sync --frozen`).
- **Full Local Regression**: 179 unit, integration, and transformer pipeline tests passing locally (`python -m pytest -q`).

---

## Documentation Suite

1. [`README.md`](../README.md): Recruiter-grade overview, quickstart, system comparison, and visual showcase.
2. [`docs/final_technical_report.md`](final_technical_report.md): 30-section research-grade engineering report with formal ablation studies and failure taxonomy.
3. [`docs/system_implementation_guide.md`](system_implementation_guide.md): Module-by-module implementation reference with architectural dataflow diagrams.
4. [`docs/api.md`](api.md): REST API reference with OpenAPI schemas and request/response payloads.
5. [`docs/demo.md`](demo.md): Visual UI and terminal CLI showcase walkthrough.
6. [`docs/canonical_architecture_manifest.md`](canonical_architecture_manifest.md): Formal frozen invariant specification.
7. [`docs/canonical_architecture_audit.md`](canonical_architecture_audit.md): Provenance and audit verification report.

---

## Known Limitations & Architectural Trade-offs

1. **Conservative Abstention Floor**: The fail-closed semantic verifier prioritizes zero hallucination over answer coverage ($68.0\%$ coverage on multi-hop benchmarks), safely abstaining to `UNKNOWN` when cross-source entity linkages are incomplete.
2. **Compositional Reasoning Depth**: Highly asymmetric multi-hop queries requiring $>3$ distinct document hops may experience partial entity drop if intermediate bridging passages receive lower reranking scores.
3. **Single Domain Scope**: Canonical evaluations are calibrated on multi-source news synthesis (MultiHopRAG); domain adaptation to biomedical or legal corpora requires regenerating BM25 and vector indices.

---

## Reproducibility Guide

```powershell
# 1. Clone repository
git clone https://github.com/mmnsrti/adaptive-agentic-rag.git
cd adaptive-agentic-rag

# 2. Setup virtual environment
uv venv .venv --python 3.12
.venv\Scripts\activate
uv sync --frozen

# 3. Launch FastAPI server
uvicorn adaptive_agentic_rag.api.app:create_app --factory --host 127.0.0.1 --port 8000

# 4. Run automated test suite
python -m pytest -q
```

