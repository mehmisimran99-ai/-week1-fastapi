"""Vector store implementations for RAG"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
import json
import os
import numpy as np


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        pass

    @abstractmethod
    def similarity_search(self, query_embedding: List[float], top_k: int) -> List[Document]:
        pass

    @abstractmethod
    def persist(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store with cosine similarity"""
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def similarity_search(self, query_embedding: List[float], top_k: int) -> List[Document]:
        if not self.embeddings:
            return []

        query = np.array(query_embedding)
        embeddings = np.array(self.embeddings)

        # Cosine similarity
        query_norm = np.linalg.norm(query)
        embed_norms = np.linalg.norm(embeddings, axis=1)
        similarities = np.dot(embeddings, query) / (embed_norms * query_norm + 1e-8)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]

    def persist(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        data = {
            "documents": [{"content": d.content, "metadata": d.metadata} for d in self.documents],
            "embeddings": self.embeddings,
        }
        with open(os.path.join(path, "vectors.json"), "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        with open(os.path.join(path, "vectors.json"), "r") as f:
            data = json.load(f)
        self.documents = [Document(**d) for d in data["documents"]]
        self.embeddings = data["embeddings"]


class ChromaDBVectorStore(VectorStore):
    """ChromaDB vector store"""
    def __init__(self, collection_name: str = "documents", persist_dir: str = "./chroma_db"):
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb not installed: pip install chromadb")

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        ids = [doc.metadata.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def similarity_search(self, query_embedding: List[float], top_k: int) -> List[Document]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = []
        if results["documents"] and results["documents"][0]:
            for i, (doc_text, metadata) in enumerate(zip(
                results["documents"][0], results["metadatas"][0]
            )):
                documents.append(Document(content=doc_text, metadata=metadata or {}))
        return documents

    def persist(self, path: str) -> None:
        # ChromaDB auto-persists
        pass

    def load(self, path: str) -> None:
        # ChromaDB auto-loads from persist_dir
        pass


def create_vector_store(store_type: str, **kwargs) -> VectorStore:
    stores = {
        "memory": InMemoryVectorStore,
        "inmemory": InMemoryVectorStore,
        "chromadb": ChromaDBVectorStore,
        "chroma": ChromaDBVectorStore,
    }
    cls = stores.get(store_type.lower())
    if not cls:
        raise ValueError(f"Unknown vector store: {store_type}")
    return cls(**kwargs)