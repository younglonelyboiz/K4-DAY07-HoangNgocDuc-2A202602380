from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        # Tokenize for keyword search (simple lowercase split)
        tokens = doc.content.lower().split()
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": embedding,
            "tokens": set(tokens)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_emb = self._embedding_fn(query)
        query_tokens = set(query.lower().split())
        
        results = []
        for r in records:
            # 1. Semantic score (cosine similarity)
            vec = r["embedding"]
            sem_score = 0.0
            if vec and query_emb:
                sem_score = _dot(query_emb, vec) / (
                    (sum(x*x for x in query_emb)**0.5) * (sum(x*x for x in vec)**0.5) + 1e-9
                )
                
            # 2. Keyword score (simple Jaccard or overlap)
            # using overlap count as a simple TF approximation
            kw_score = len(query_tokens.intersection(r["tokens"])) / (len(query_tokens) + 1e-9)
            
            # Hybrid score (RRF-like or simple weighted sum)
            # Give semantic search 70% weight, keyword 30%
            hybrid_score = 0.7 * sem_score + 0.3 * kw_score
            
            results.append({
                "id": r["id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "score": hybrid_score
            })
            
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.
        """
        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.
        """
        if not metadata_filter:
            return self.search(query, top_k)
            
        filtered_records = []
        for r in self._store:
            match = True
            for k, v in metadata_filter.items():
                if r["metadata"].get(k) != v:
                    match = False
                    break
            if match:
                filtered_records.append(r)
                
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.
        """
        initial_size = len(self._store)
        self._store = [r for r in self._store if r.get("id") != doc_id]
        return len(self._store) < initial_size
