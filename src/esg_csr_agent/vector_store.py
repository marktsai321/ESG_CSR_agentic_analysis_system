from __future__ import annotations

"""
Abstracted vector store interface.

Swap the backend by changing VECTOR_STORE_BACKEND in config / .env.
Currently supported: "chromadb".
"""

import abc
from typing import Any

import esg_csr_agent.config as cfg


class VectorStore(abc.ABC):
    """Abstract base class — all backends implement this interface."""

    @abc.abstractmethod
    def add_documents(
        self,
        namespace: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Store chunks + embeddings under *namespace*. Return count stored."""

    @abc.abstractmethod
    def query(
        self,
        namespace: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return top-k results as list of {text, metadata, score}."""

    @abc.abstractmethod
    def namespace_exists(self, namespace: str) -> bool:
        """Check whether a namespace already has documents."""

    @abc.abstractmethod
    def delete_namespace(self, namespace: str) -> None:
        """Remove all data for a namespace."""


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed implementation (local, zero infrastructure)."""

    def __init__(self, persist_dir: str | None = None):
        import chromadb

        path = persist_dir or str(cfg.VECTOR_STORE_DIR)
        self._client = chromadb.PersistentClient(path=path)

    def _get_collection(self, namespace: str):
        return self._client.get_or_create_collection(
            name=namespace,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        namespace: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        col = self._get_collection(namespace)
        ids = [f"{namespace}_{i}" for i in range(len(chunks))]
        col.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(chunks),
        )
        return len(chunks)

    def query(
        self,
        namespace: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        col = self._get_collection(namespace)
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            out.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "score": 1.0 - results["distances"][0][i],
            })
        return out

    def namespace_exists(self, namespace: str) -> bool:
        try:
            col = self._client.get_collection(namespace)
            return col.count() > 0
        except Exception:
            return False

    def delete_namespace(self, namespace: str) -> None:
        try:
            self._client.delete_collection(namespace)
        except Exception:
            pass


def get_vector_store(**kwargs) -> VectorStore:
    """Factory: return the configured backend instance."""
    backend = cfg.VECTOR_STORE_BACKEND.lower()
    if backend == "chromadb":
        return ChromaVectorStore(**kwargs)
    raise ValueError(f"Unsupported vector store backend: {backend!r}")
