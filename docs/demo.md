# Interactive API Demo & Showcase Guide

- **Target Audience**: AI Engineers, Recruiters, and Technical Reviewers
- **Service**: Adaptive Agentic RAG FastAPI Inference Service
- **Interactive OpenAPI Playground**: `http://127.0.0.1:8000/docs`
- **Demo Script**: [`scripts/demo_api.py`](../scripts/demo_api.py)

---

## 1. What the Demo Showcases

This interactive demo exercises the live **Adaptive Agentic RAG** service across three realistic production scenarios:

1. **Scenario A: Answered Multi-Hop Query with Inline Citations**:
   Demonstrates adaptive routing, hybrid retrieval (Dense + BM25 via Reciprocal Rank Fusion), cross-encoder reranking, single-pass generation, DeBERTa NLI claim grounding, and structured citation metadata.
2. **Scenario B: Safe Engineering Observability (Trace Mode)**:
   Demonstrates structured runtime telemetry (`include_trace = true`) exposing execution strategy, candidate count, and citation validity flags without leaking internal chain-of-thought tokens or prompts.
3. **Scenario C: Fail-Closed Safe Abstention**:
   Demonstrates how the system safely refuses to hallucinate on unanswerable or out-of-corpus queries, returning `HTTP 200 OK` with `direct_answer: "UNKNOWN"`, `abstained: true`, and zero fabricated citations.

---

## 2. Launching the API Service

### Step 1: Activate the Environment
```powershell
# Windows PowerShell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
& d:\app\ai\adaptive-agentic-rag\.venv\Scripts\Activate.ps1
```

### Step 2: Start the Uvicorn ASGI Server
```powershell
uvicorn adaptive_agentic_rag.api.app:create_app --factory --host 127.0.0.1 --port 8000
```
*Note*: During startup, the FastAPI `lifespan` handler pre-warms all transformer models (`Qwen3-Embedding-0.6B`, `BAAI/bge-reranker-base`, `Qwen2.5-1.5B-Instruct`, and `cross-encoder/nli-deberta-v3-small`) into GPU VRAM and verifies local Qdrant collection `multihop_chunks_v2`.

---

## 3. Running the Automated Demo Script

In a second terminal, execute:
```powershell
python scripts/demo_api.py
```

### Expected Output Walkthrough

```text
================================================================================
          ADAPTIVE AGENTIC RAG — FASTAPI DEMO & SHOWCASE
================================================================================

[1/5] Liveness Check: GET /health
  -> Status: 200 OK | Process Healthy: True

[2/5] Readiness Probe: GET /ready
  -> Status: 200 OK | Pipeline Loaded: True | Collection: multihop_chunks_v2

[3/5] System Architecture: GET /v1/system
  -> Architecture Version: V2-A (Frozen Canonical)
  -> Embedding Model: Qwen/Qwen3-Embedding-0.6B (1024-dim)
  -> Reranker Model: BAAI/bge-reranker-base
  -> Generator Model: Qwen/Qwen2.5-1.5B-Instruct
  -> NLI Model: cross-encoder/nli-deberta-v3-small
  -> Total Corpus Chunks: 8,173

--------------------------------------------------------------------------------
[4/5] Scenario A: Answered Multi-Hop Query
Query: 'Did Sam Altman return as CEO of OpenAI according to news reports?'
  -> Direct Answer: Yes
  -> Abstained: False
  -> HTTP Request Timing: 589.67 ms
  -> Answer: Yes, Sam Altman returned as CEO of OpenAI with a new initial board [1].
  -> Verified Citations (1):
     [1] The Verge: "Sam Altman returns to OpenAI as CEO"
         URL: https://www.theverge.com/2023/11/22/sam-altman-returns-openai-ceo
         Entailment Score: 0.9412

--------------------------------------------------------------------------------
[5/5] Scenario B: Trace Mode (Observability)
Query: 'Compare the reports on OpenAI leadership changes.'
  -> Route: complex (hybrid retrieval strategy)
  -> Retrived Candidates: 10
  -> Citation Validity: True

--------------------------------------------------------------------------------
[6/5] Scenario C: Fail-Closed Safe Abstention
Query: 'Which country won the 2038 FIFA World Cup tournament?'
  -> Direct Answer: UNKNOWN
  -> Abstained: True (Safe Fail-Closed Gating)
  -> Citations: 0 (Zero Hallucinated Citations)
  -> Status Code: 200 OK

================================================================================
All demo scenarios executed successfully!
================================================================================
```

---

## 4. Interactive Swagger & OpenAPI UI

Once the server is running, navigate to:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

You can interactively execute queries and inspect JSON response schemas directly from your browser.

