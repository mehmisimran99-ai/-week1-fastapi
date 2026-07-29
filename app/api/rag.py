"""RAG API endpoints"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os

from app.rag.pipeline import RAGPipeline, create_rag_pipeline, RAGResponse
from app.rag.vector_store import Document
from app.config import config

router = APIRouter(prefix="/rag", tags=["RAG"])

# Global pipeline instance
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = create_rag_pipeline(config.rag)
    return _pipeline


class IngestRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[dict]] = None


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str


@router.post("/ingest", response_model=dict)
async def ingest_documents(request: IngestRequest):
    """Ingest text documents into the RAG system"""
    pipeline = get_pipeline()
    documents = []
    for i, text in enumerate(request.texts):
        meta = request.metadatas[i] if request.metadatas and i < len(request.metadatas) else {}
        meta["id"] = meta.get("id", f"doc_{i}")
        documents.append(Document(content=text, metadata=meta))

    count = pipeline.ingest(documents)
    return {"ingested_chunks": count}


@router.post("/ingest/file", response_model=dict)
async def ingest_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    """Upload and ingest a file (txt, pdf)"""
    pipeline = get_pipeline()

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from app.rag.pipeline import DocumentLoader
        documents = DocumentLoader.load_file(tmp_path)

        if metadata:
            import json
            meta = json.loads(metadata)
            for doc in documents:
                doc.metadata.update(meta)

        count = pipeline.ingest(documents)
        return {"ingested_chunks": count, "source": file.filename}
    finally:
        os.unlink(tmp_path)


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system"""
    pipeline = get_pipeline()
    try:
        response: RAGResponse = pipeline.query(request.question)
        return QueryResponse(
            answer=response.answer,
            sources=[{"content": d.content, "metadata": d.metadata} for d in response.sources],
            query=response.query,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/persist")
async def persist_index(path: str = "data/vector_store"):
    """Persist the vector store to disk"""
    pipeline = get_pipeline()
    pipeline.persist(path)
    return {"status": "persisted", "path": path}


@router.post("/load")
async def load_index(path: str = "data/vector_store"):
    """Load the vector store from disk"""
    global _pipeline
    _pipeline = create_rag_pipeline(config.rag)
    _pipeline.load(path)
    return {"status": "loaded", "path": path}