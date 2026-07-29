"""Application configuration"""
from pydantic import BaseModel
import os


class RAGConfig(BaseModel):
    embedding_model: str = "nomic-embed-text"
    vector_store: str = "memory"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    llm_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"


class AppConfig(BaseModel):
    rag: RAGConfig = RAGConfig()
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            rag=RAGConfig(
                embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
                vector_store=os.getenv("VECTOR_STORE", "memory"),
                chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
                top_k=int(os.getenv("TOP_K", "4")),
                llm_model=os.getenv("LLM_MODEL", "llama3.1:8b"),
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "true").lower() == "true",
        )


config = AppConfig.from_env()