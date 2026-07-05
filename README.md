# AI Research Agent

[![CI](https://github.com/parikhrjt/ai-research-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/parikhrjt/ai-research-agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-grade Retrieval-Augmented Generation (RAG) system that ingests heterogeneous research documents, indexes them in a vector database, and answers questions with inline citations.

Built as a portfolio project demonstrating end-to-end AI engineering: document parsing, intelligent chunking, local embeddings, vector search, LangGraph orchestration, and a dual FastAPI + Streamlit interface.

> **Live demo:** Deploy the API on [Render](#deploy-api-on-render) or [Railway](#deploy-api-on-railway), then host the UI on [Streamlit Cloud](#deploy-ui-on-streamlit-cloud). See the [Demo walkthrough](#demo) below.

---

## Problem Statement

Research teams accumulate knowledge across PDFs, markdown wikis, meeting notes, and CSV exports. Finding specific information requires manually searching disparate sources — slow, error-prone, and unscalable.

This agent solves that by:

1. **Ingesting** PDF, TXT, Markdown, and CSV files into a unified index
2. **Retrieving** the most relevant passages via semantic vector search
3. **Generating** grounded answers with numbered inline citations
4. **Running locally** with free embedding models and optional Ollama LLMs — no API keys required

---

## Features

| Capability | Details |
|---|---|
| **Multi-format ingestion** | PDF (pypdf), Markdown (front-matter aware), TXT, CSV (row-to-narrative) |
| **Smart chunking** | Header-aware Markdown splits, CSV row batching, recursive overlap chunking |
| **Local embeddings** | `all-MiniLM-L6-v2` via sentence-transformers (384-dim, free) |
| **Vector stores** | PostgreSQL + pgvector (default) or ChromaDB (zero-config local) |
| **RAG pipeline** | LangGraph two-node graph: retrieve → generate |
| **Cited answers** | Numbered sources `[1]`, `[2]` with excerpt + relevance score |
| **LLM providers** | Ollama (local, default) · OpenAI (optional via `.env`) |
| **API** | FastAPI with `/health`, `/upload`, `/ask`, `/documents` |
| **UI** | Streamlit frontend for upload + Q&A |
| **Observability** | Structured logging (structlog), health checks, error hierarchy |
| **Deployment** | Docker Compose · Render · Railway · Streamlit Cloud |
| **CI** | GitHub Actions — pytest on every push |

---

## Architecture

```mermaid
flowchart TB
    subgraph Client Layer
        UI[Streamlit UI<br/>:8501]
        API[FastAPI<br/>:8000]
    end

    subgraph Ingestion Pipeline
        UPLOAD[File Upload] --> PARSE[Format Parsers<br/>PDF · MD · TXT · CSV]
        PARSE --> CHUNK[Smart Chunker<br/>header · row · recursive]
        CHUNK --> EMBED[Embedding Model<br/>all-MiniLM-L6-v2]
    end

    subgraph Storage
        EMBED --> VDB[(Vector Store<br/>pgvector / ChromaDB)]
        META[(Document Metadata<br/>UUID · hash · timestamps)]
    end

    subgraph RAG Pipeline
        Q[User Question] --> QEMB[Query Embedding]
        QEMB --> SEARCH[Similarity Search<br/>top-K + threshold]
        SEARCH --> CTX[Context Builder<br/>numbered citations]
        CTX --> LLM[LLM Generation<br/>Ollama / OpenAI]
        LLM --> ANS[Cited Answer]
    end

    UI -->|HTTP| API
    API --> UPLOAD
    API --> Q
    VDB --> SEARCH
    VDB --- META

    subgraph Infrastructure
        DB[(PostgreSQL 16<br/>+ pgvector)]
        OLLAMA[Ollama<br/>llama3.2]
    end

    VDB -.-> DB
    LLM -.-> OLLAMA
```

### Request Flow

```
Upload:  file → parse → chunk → embed → upsert → response
Ask:     question → embed → retrieve → context → LLM → cited answer
```

---

## Project Structure

```
ai-research-agent/
├── app/
│   ├── api/              # FastAPI routes and schemas
│   ├── chunking/         # Format-aware text splitting
│   ├── core/             # Config, logging, exceptions
│   ├── embeddings/       # sentence-transformers provider
│   ├── ingestion/        # Parsers and upload pipeline
│   ├── llm/              # Ollama and OpenAI providers
│   ├── rag/              # LangGraph pipeline + citations
│   ├── ui/               # Streamlit frontend
│   └── vectorstore/      # pgvector and ChromaDB backends
├── data/samples/         # Sample documents for demo
├── docker/               # PostgreSQL init scripts
├── scripts/              # PDF generator and seed utility
├── tests/                # Unit tests (pytest)
├── docker-compose.yml
├── render.yaml           # Render Blueprint (one-click API deploy)
├── railway.toml          # Railway config
├── streamlit_app.py      # Streamlit Cloud entry point
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Demo

Try the API in under 2 minutes — locally or against your deployed URL.

```bash
# Set your API base (local or cloud)
export API_URL=http://localhost:8000
# export API_URL=https://ai-research-api.onrender.com
```

### Step 1 — Health check

```bash
curl -s "$API_URL/health" | python3 -m json.tool
```

Expected: `"status": "healthy"` or `"degraded"` (degraded is fine if Ollama isn't running locally).

### Step 2 — Upload a sample document

```bash
curl -s -X POST "$API_URL/upload" \
  -F "file=@data/samples/transformer_architecture.md" | python3 -m json.tool
```

Expected response:

```json
{
  "document_id": "abc123...",
  "filename": "transformer_architecture.md",
  "file_type": "md",
  "chunk_count": 8,
  "char_count": 2303,
  "message": "Document ingested successfully"
}
```

### Step 3 — List indexed documents

```bash
curl -s "$API_URL/documents" | python3 -m json.tool
```

### Step 4 — Ask a question with citations

```bash
curl -s -X POST "$API_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What BLEU score did the Transformer achieve on English-to-German translation?"}' \
  | python3 -m json.tool
```

Expected: an answer with inline `[1]` citations and a `citations` array with filename, excerpt, and relevance score.

### Step 5 — Upload more formats

```bash
# Plain text notes
curl -s -X POST "$API_URL/upload" -F "file=@data/samples/rag_pipeline_notes.txt"

# CSV experiment data
curl -s -X POST "$API_URL/upload" -F "file=@data/samples/experiment_results.csv"

# PDF research brief
curl -s -X POST "$API_URL/upload" -F "file=@data/samples/research_brief.pdf"
```

### Sample questions

| Question | Expected source |
|---|---|
| "What BLEU score did the Transformer achieve?" | `transformer_architecture.md` |
| "What chunking strategies are used for CSV files?" | `rag_pipeline_notes.txt` |
| "What was the RAG answer relevance score for EXP-005?" | `experiment_results.csv` |
| "What are the key findings about RAG systems?" | `research_brief.pdf` |

### One-liner smoke test

```bash
API_URL=http://localhost:8000 \
  curl -sf "$API_URL/health" && \
  curl -sf -X POST "$API_URL/upload" -F "file=@data/samples/transformer_architecture.md" && \
  curl -sf -X POST "$API_URL/ask" -H "Content-Type: application/json" \
    -d '{"question": "What is self-attention?"}' && \
  echo "✓ All endpoints OK"
```

---

## Cloud Deployment

Cloud deployments use **ChromaDB** (no PostgreSQL setup) and **OpenAI** for answer generation (Ollama isn't available on Render/Railway free tiers). Embeddings still run locally via sentence-transformers.

> **Note:** Free-tier disks are ephemeral — uploaded documents reset on redeploy. Fine for demos; use persistent storage for production.

### Deploy API on Render

1. Fork/clone this repo to your GitHub account
2. Go to [render.com](https://render.com) → **New** → **Blueprint** → connect `parikhrjt/ai-research-agent`
3. Render detects `render.yaml` automatically
4. Add **`OPENAI_API_KEY`** in the Render dashboard when prompted
5. Deploy — your API will be at `https://ai-research-api.onrender.com` (or similar)

Manual deploy (without Blueprint):

| Setting | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build command** | `pip install -r requirements.txt` |
| **Start command** | `sh scripts/start_api.sh` |
| **Health check** | `/health` |

Environment variables — see [`.env.cloud.example`](.env.cloud.example).

### Deploy API on Railway

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Select `parikhrjt/ai-research-agent`
3. Railway reads `railway.toml` automatically
4. Add environment variable: **`OPENAI_API_KEY`** = your key
5. Generate a public domain under **Settings → Networking**
6. Copy the URL (e.g. `https://ai-research-agent-production.up.railway.app`)

### Deploy UI on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Connect GitHub repo `parikhrjt/ai-research-agent`
3. Configure:

| Setting | Value |
|---|---|
| **Branch** | `main` |
| **Main file path** | `streamlit_app.py` |
| **App URL** | `ai-research-agent` (or your choice) |

4. Under **Advanced settings → Secrets**, add:

```toml
API_BASE_URL = "https://your-api-url.onrender.com"
```

Replace with your Render or Railway API URL from the steps above.

5. Click **Deploy** — UI will be at `https://ai-research-agent.streamlit.app`

### Post-deploy checklist

- [ ] API `/health` returns 200
- [ ] Upload a sample doc via Streamlit or `curl`
- [ ] Ask a question and verify citations appear
- [ ] Add live URLs to your GitHub repo **About** section and resume

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- (Optional) Python 3.11+ for local development

### 1. Clone and configure

```bash
cd ai-research-agent
cp .env.example .env
```

### 2. Generate sample PDF (optional)

```bash
pip install fpdf2
python scripts/generate_sample_pdf.py
```

### 3. Launch the stack

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

On first run, the `ollama-pull` service downloads `llama3.2` (~2 GB). The API may show `degraded` until the model is ready.

### 4. Seed sample documents (optional)

```bash
# With the stack running:
docker compose exec api python scripts/seed_samples.py
```

---

## Local Development (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Use ChromaDB to skip PostgreSQL
export VECTOR_STORE=chroma

# Start API
uvicorn app.api.main:app --reload --port 8000

# Start UI (separate terminal)
export API_BASE_URL=http://localhost:8000
streamlit run app/ui/streamlit_app.py
```

For full local setup with pgvector, start PostgreSQL with the pgvector extension and set `DATABASE_URL` in `.env`. For LLM answers, run [Ollama](https://ollama.ai) locally: `ollama pull llama3.2`.

---

## API Reference

### `GET /health`

Returns system status and component health.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "vector_store": "pgvector",
  "llm_provider": "ollama",
  "components": {
    "vector_store": true,
    "llm": true
  }
}
```

### `POST /upload`

Upload and ingest a document.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data/samples/transformer_architecture.md"
```

```json
{
  "document_id": "a1b2c3d4-...",
  "filename": "transformer_architecture.md",
  "file_type": "md",
  "chunk_count": 12,
  "char_count": 2847,
  "message": "Document ingested successfully"
}
```

### `POST /ask`

Ask a question against indexed documents.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What BLEU score did the Transformer achieve?"}'
```

```json
{
  "answer": "The Transformer achieved a BLEU score of 28.4 on the WMT 2014 English-to-German translation task [1].",
  "citations": [
    {
      "index": 1,
      "filename": "transformer_architecture.md",
      "excerpt": "On the WMT 2014 English-to-German translation task, the base Transformer achieved 28.4 BLEU...",
      "score": 0.82,
      "chunk_index": 3
    }
  ],
  "retrieved_count": 5
}
```

### `GET /documents`

List all indexed documents.

```bash
curl http://localhost:8000/documents
```

---

## Configuration

All settings are managed via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `VECTOR_STORE` | `pgvector` | `pgvector` or `chroma` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OLLAMA_MODEL` | `llama3.2` | Local LLM model name |
| `OPENAI_API_KEY` | _(empty)_ | Required only for OpenAI provider |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per query |

---

## Testing

```bash
# Run all tests (uses ChromaDB in /tmp, no external services)
PYTHONPATH=. pytest tests/ -v

# With coverage
PYTHONPATH=. pytest tests/ --cov=app --cov-report=term-missing
```

CI runs automatically on every push to `main` via [GitHub Actions](.github/workflows/ci.yml).

---

## Switching to OpenAI

```bash
# In .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Embeddings remain local (free). Only answer generation uses the OpenAI API.

---

## Future Improvements

- **Hybrid retrieval** — combine BM25 sparse search with dense embeddings
- **Cross-encoder re-ranking** — re-score top-20 candidates before generation
- **Async ingestion queue** — Celery + Redis for large batch uploads
- **Evaluation harness** — automated RAGAS metrics (faithfulness, relevance, precision)
- **Multi-tenant isolation** — per-user document namespaces and access control
- **Conversation memory** — multi-turn Q&A with session context
- **Observability** — OpenTelemetry tracing, Prometheus metrics, LangSmith integration
- **Document deduplication** — content-hash based skip on re-upload

---

## License

MIT
