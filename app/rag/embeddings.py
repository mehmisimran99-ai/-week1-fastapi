"""Embedding providers for RAG"""
from abc import ABC, abstractmethod
from typing import List
import httpx


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass


class OllamaEmbeddings(EmbeddingProvider):
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        resp = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self.model = model
        self.api_key = api_key

    def embed(self, texts: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Local embeddings using sentence-transformers (no API needed)"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]


def create_embedding_provider(provider: str, **kwargs) -> EmbeddingProvider:
    providers = {
        "ollama": OllamaEmbeddings,
        "openai": OpenAIEmbeddings,
        "sentence-transformers": SentenceTransformerEmbeddings,
        "local": SentenceTransformerEmbeddings,
    }
    cls = providers.get(provider.lower())
    if not cls:
        raise ValueError(f"Unknown embedding provider: {provider}")
    return cls(**kwargs)