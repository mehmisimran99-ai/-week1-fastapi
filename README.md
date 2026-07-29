# LLM Engineering Portfolio

A production-ready **RAG pipeline** + **tool-calling agent** with **evaluation harness** — built to demonstrate LLM engineering skills for job applications.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- Docker (for containerized deployment)

### Local Development

```bash
# 1. Start Ollama and pull models
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run API
uvicorn app.main:app --reload --port 8000

# 5. Test endpoints
# Health check
curl http://localhost:8000/health

# Ingest documents
curl -X POST http://localhost:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["FastAPI is a modern Python web framework", "Ollama runs LLMs locally"], "metadatas": [{"source": "docs"}, {"source": "blog"}]}'

# Query RAG
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?"}'

# Chat with agent
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Calculate 15 * 23 + 100"}'

# Stream agent response
curl -X POST http://localhost:8000/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Write Python code to compute fibonacci(10)"}'

# Run evaluation suite
python scripts/run_eval.py
```

### Docker

```bash
# Build
docker build -t llm-portfolio .

# Run (requires Ollama on host)
docker run -p 8000:8000 --add-host=host.docker.internal:host-gateway llm-portfolio
```

### Fly.io (Free Tier)

```bash
# Install flyctl
winget install Flyctl  # Windows
# brew install flyctl    # Mac

# Deploy
fly auth login
fly launch --no-deploy
fly deploy
```

## 📁 Project Structure

```
llm-portfolio/
├── app/
│   ├── main.py              # FastAPI app + routers
│   ├── config.py            # Pydantic settings
│   ├── docs.py              # Custom OpenAPI schema
│   ├── api/
│   │   ├── rag.py           # RAG endpoints (/rag/*)
│   │   ├── agent.py         # Agent endpoints (/agent/*)
│   │   └── eval.py          # Evaluation endpoints (/eval/*)
│   ├── rag/
│   │   ├── pipeline.py      # RAG orchestration
│   │   ├── embeddings.py    # Ollama/OpenAI/local embeddings
│   │   ├── vector_store.py  # In-memory/ChromaDB stores
│   │   └── llm.py           # LLM providers
│   ├── agent/
│   │   ├── state.py         # Agent config/state
│   │   ├── tools.py         # Calculator, Python, Web, Files
│   │   └── runner.py        # LangGraph-style agent loop
│   └── eval/
│       └── harness.py       # Evaluation framework
├── scripts/
│   └── run_eval.py          # CLI evaluation runner
├── tests/
│   └── test_main.py         # Health/echo tests
├── requirements.txt
├── Dockerfile
├── fly.toml
└── .github/workflows/ci.yml
```

## 🔧 API Endpoints

### RAG (`/rag`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/rag/ingest` | Ingest text documents |
| POST | `/rag/ingest/file` | Upload PDF/TXT/MD files |
| POST | `/rag/query` | Query knowledge base |
| POST | `/rag/persist` | Save vector index |
| POST | `/rag/load` | Load vector index |

### Agent (`/agent`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/agent/chat` | Non-streaming chat |
| POST | `/agent/chat/stream` | SSE streaming chat |
| GET | `/agent/tools` | List available tools |
| POST | `/agent/configure` | Update agent config |

### Evaluation (`/eval`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/eval/run` | Start evaluation run |
| GET | `/eval/run/{run_id}` | Get results |
| GET | `/eval/runs` | List all runs |
| GET | `/eval/suites` | List test suites |

## 🛠️ Built-in Tools

| Tool | Description | Example |
|------|-------------|---------|
| `calculator` | Safe math evaluation | `"Calculate sqrt(144) + 5^2"` |
| `python_execute` | Run Python code | `"Write code to compute fibonacci(10)"` |
| `file_read` | Read local files | `"Read requirements.txt"` |
| `file_write` | Write local files | `"Create hello.py with print('hi')"` |
| `web_search` | Tavily web search | `"Search for latest AI news"` |

## 📊 Evaluation Framework

Run comprehensive tests:
```bash
# CLI (used in CI)
python scripts/run_eval.py

# Or via API
curl -X POST http://localhost:8000/eval/run \
  -H "Content-Type: application/json" \
  -d '{"suite": "full"}'
```

**Test Suites:**
- **Math** - Arithmetic, calculus, algebra
- **Reasoning** - Logic, general knowledge
- **Tools** - Calculator, Python execution
- **Full** - All of the above

**Evaluators:**
- Exact match
- Substring containment
- LLM-as-judge (configurable)
- Tool usage verification

## 🐳 Docker

```dockerfile
# Multi-stage build for small image (~200MB)
FROM python:3.11-slim as builder
# ... install deps

FROM python:3.11-slim
# ... copy from builder, run as non-root
```

## ⚙️ Configuration

Environment variables (or `.env`):
```bash
# LLM
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.1

# RAG
EMBEDDING_MODEL=nomic-embed-text
VECTOR_STORE=memory  # or chromadb
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=4

# Agent
AGENT_MAX_ITERATIONS=5

# External APIs
TAVILY_API_KEY=your_key_here  # For web search
OPENAI_API_KEY=your_key_here  # Alternative to Ollama
```

## 📈 CI/CD Pipeline

GitHub Actions (`.github/workflows/ci.yml`):
1. **Lint** - `ruff check .`
2. **Test** - `pytest -v`
3. **Build** - `docker build`
4. **Eval** - `python scripts/run_eval.py` (must pass ≥80%)
5. **Deploy** - Auto-deploy to Fly.io on main branch

## 🎯 Portfolio Talking Points

| Area | What This Demonstrates |
|------|------------------------|
| **RAG** | End-to-end: ingestion → chunking → embedding → retrieval → generation → persistence |
| **Agent** | Tool calling, streaming, multi-turn, config-driven behavior |
| **Eval** | Test suites, multiple evaluators, CI gates, JSON reports |
| **Infra** | Docker multi-stage, Fly.io GPU/CPU, health checks, observability |
| **Code Quality** | Type hints, Pydantic, modular architecture, async throughout |

## 📝 License

MIT — use freely for learning and portfolio purposes.