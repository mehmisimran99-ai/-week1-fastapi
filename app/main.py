from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import rag, agent, eval
from app.docs import custom_openapi
from app.config import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting RAG API on {config.host}:{config.port}")
    print(f"RAG Config: {config.rag.model_dump()}")
    # Customize OpenAPI
    custom_openapi(app)
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="LLM Engineering Portfolio - RAG & Agent API",
    description="Production RAG pipeline with Ollama embeddings + LLM, plus Agent with tool calling",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rag.router)
app.include_router(agent.router)
app.include_router(eval.router)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "rag-api"}


@app.get("/echo")
async def echo(message: str = "hello"):
    return {"message": message}


@app.get("/")
async def root():
    return {
        "service": "RAG & Agent API",
        "version": "0.3.0",
        "docs": "/docs",
        "endpoints": {
            "rag_ingest": "/rag/ingest",
            "rag_ingest_file": "/rag/ingest/file",
            "rag_query": "/rag/query",
            "rag_persist": "/rag/persist",
            "rag_load": "/rag/load",
            "agent_chat": "/agent/chat",
            "agent_stream": "/agent/chat/stream",
            "agent_tools": "/agent/tools",
            "eval_run": "/eval/run",
            "eval_result": "/eval/run/{run_id}",
            "eval_runs": "/eval/runs",
            "eval_suites": "/eval/suites",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)