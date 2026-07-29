"""RAG Pipeline - orchestrates document loading, embedding, retrieval, and generation"""
from typing import List, Optional
from dataclasses import dataclass
import os

from app.rag.vector_store import (
    VectorStore, Document, create_vector_store
)
from app.llm.client import create_llm_client, LLMClient, LLMConfig
from app.config import RAGConfig


@dataclass
class RAGResponse:
    answer: str
    sources: List[Document]
    query: str


class DocumentLoader:
    @staticmethod
    def load_text(text: str, metadata: Optional[dict] = None) -> Document:
        return Document(content=text, metadata=metadata or {})

    @staticmethod
    def load_file(path: str) -> List[Document]:
        """Load documents from file (txt, md, pdf)"""
        ext = os.path.splitext(path)[1].lower()

        if ext in [".txt", ".md", ".py", ".js", ".json", ".yaml", ".yml"]:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return [Document(content=content, metadata={"source": path, "type": ext[1:]})]

        elif ext == ".pdf":
            try:
                import pdfplumber
                docs = []
                with pdfplumber.open(path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text:
                            docs.append(Document(
                                content=text,
                                metadata={"source": path, "page": i + 1, "type": "pdf"}
                            ))
                return docs
            except ImportError:
                raise ImportError("pdfplumber required for PDF: pip install pdfplumber")

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def chunk_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Split documents into overlapping chunks"""
        chunks = []
        for doc in documents:
            text = doc.content
            if len(text) <= chunk_size:
                chunks.append(doc)
                continue

            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end]
                chunk_meta = doc.metadata.copy()
                chunk_meta["chunk_index"] = len(chunks)
                chunk_meta["parent_id"] = doc.metadata.get("id", "")
                chunks.append(Document(content=chunk_text, metadata=chunk_meta))
                start += chunk_size - chunk_overlap
        return chunks


class RAGPipeline:
    def __init__(
        self,
        config: RAGConfig,
        vector_store: Optional[VectorStore] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.config = config
        self.vector_store = vector_store or create_vector_store(config.vector_store)
        self.llm_client = llm_client or create_llm_client(LLMConfig(
            provider="ollama",
            model=config.llm_model,
            base_url=config.ollama_base_url,
        ))
        self.embedding_model = config.embedding_model

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from Ollama"""
        import httpx
        import asyncio

        async def _embed():
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.config.ollama_base_url}/api/embed",
                    json={"model": self.embedding_model, "input": texts}
                )
                resp.raise_for_status()
                return resp.json()["embeddings"]

        return asyncio.run(_embed())

    def ingest(self, documents: List[Document]) -> int:
        """Ingest documents: chunk, embed, store"""
        # Chunk documents
        chunks = DocumentLoader.chunk_documents(
            documents, self.config.chunk_size, self.config.chunk_overlap
        )

        # Get embeddings
        texts = [c.content for c in chunks]
        embeddings = self._get_embeddings(texts)

        # Store
        self.vector_store.add_documents(chunks, embeddings)
        return len(chunks)

    def query(self, question: str) -> RAGResponse:
        """Query the RAG system: retrieve -> generate"""
        # Embed query
        query_embedding = self._get_embeddings([question])[0]

        # Retrieve
        sources = self.vector_store.similarity_search(query_embedding, self.config.top_k)

        # Build context
        context = "\n\n".join([
            f"[Source {i+1}: {s.metadata.get('source', 'unknown')}]\n{s.content}"
            for i, s in enumerate(sources)
        ])

        # Generate
        prompt = f"""Answer the question based on the provided context. Cite sources using [Source X].

Context:
{context}

Question: {question}

Answer:"""

        system = "You are a helpful assistant that answers questions using only the provided context. If the context doesn't contain the answer, say so."
        answer = self.llm_client.generate(prompt, system_prompt=system)

        return RAGResponse(answer=answer, sources=sources, query=question)

    def persist(self, path: str):
        self.vector_store.persist(path)

    def load(self, path: str):
        self.vector_store.load(path)


def create_rag_pipeline(config: RAGConfig) -> RAGPipeline:
    return RAGPipeline(config)