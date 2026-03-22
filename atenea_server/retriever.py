from typing import List, Optional, Dict, Any
import logging
from .embedder import Embedder
from .vector_store import VectorStore
from .bm25_index import BM25Index

logger = logging.getLogger(__name__)


class Retriever:
    """
    Hybrid retriever combining semantic vector search with BM25 keyword search.

    The hybrid approach improves retrieval quality by:
    - Using vector search for semantic similarity (captures meaning)
    - Using BM25 for exact keyword matches (captures specific symbols, names)
    - Combining scores with configurable weights
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore,
                 vector_weight: float = 0.7, bm25_weight: float = 0.3):
        """
        Initialize the hybrid retriever.

        Args:
            embedder: Embedder instance for generating query embeddings
            vector_store: VectorStore instance for semantic search
            vector_weight: Weight for vector search scores (default: 0.7)
            bm25_weight: Weight for BM25 search scores (default: 0.3)
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        # BM25 index per collection
        self._bm25_indices: Dict[str, BM25Index] = {}
        self._index_initialized: Dict[str, bool] = {}

    def _get_bm25_index(self, collection_name: Optional[str] = None) -> BM25Index:
        """Get or create BM25 index for a collection."""
        target = collection_name or self.vector_store.default_collection

        if target not in self._bm25_indices:
            self._bm25_indices[target] = BM25Index()
            self._index_initialized[target] = False

        # Lazily build index from vector store if not initialized
        if not self._index_initialized.get(target, False):
            logger.info(f"Building BM25 index for collection '{target}'...")
            self._bm25_indices[target].build_from_vector_store(
                self.vector_store, collection_name=target
            )
            self._index_initialized[target] = True

        return self._bm25_indices[target]

    def invalidate_bm25_index(self, collection_name: Optional[str] = None):
        """
        Invalidate the BM25 index for a collection, forcing rebuild on next search.

        Call this after indexing new documents.
        """
        target = collection_name or self.vector_store.default_collection
        if target in self._index_initialized:
            self._index_initialized[target] = False
            if target in self._bm25_indices:
                self._bm25_indices[target].clear()

    async def retrieve(self, query: str, limit: int = 20,
                       collection_name: Optional[str] = None,
                       use_hybrid: bool = True) -> List[dict]:
        """
        Retrieve relevant code chunks using hybrid search.

        Args:
            query: Search query
            limit: Maximum number of results to return
            collection_name: Optional collection to search in
            use_hybrid: If True, use hybrid search. If False, use vector-only search.

        Returns:
            List of matching chunks with their payloads
        """
        # 1. Vector search
        query_embeddings = await self.embedder.embed([query], raise_on_error=False)

        vector_results = []
        if query_embeddings:
            # Request more results to allow for fusion and deduplication
            raw_vector_results = self.vector_store.search(
                query_embeddings[0],
                limit=limit * 3,
                collection_name=collection_name
            )
            # Normalize vector scores (cosine similarity is already 0-1)
            for i, res in enumerate(raw_vector_results):
                # Qdrant returns results sorted by score, we can estimate relative scores
                # Using rank-based scoring for fusion
                score = 1.0 / (i + 1)  # Reciprocal rank
                vector_results.append((self._get_doc_key(res), score, res))

        if not use_hybrid or not vector_results:
            # Fallback to vector-only results
            return self._deduplicate_results(
                [r[2] for r in vector_results], limit
            )

        # 2. BM25 search
        bm25_index = self._get_bm25_index(collection_name)
        bm25_results = bm25_index.search(query, limit=limit * 3)

        # Normalize BM25 scores using reciprocal rank
        normalized_bm25 = []
        for i, (doc_id, score, payload) in enumerate(bm25_results):
            normalized_bm25.append((self._get_doc_key(payload), 1.0 / (i + 1), payload))

        # 3. Reciprocal Rank Fusion (RRF)
        fused_results = self._reciprocal_rank_fusion(
            vector_results, normalized_bm25, k=60
        )

        # 4. Deduplicate and limit
        return self._deduplicate_results(fused_results, limit)

    def _get_doc_key(self, payload: dict) -> str:
        """Generate a unique key for a document/chunk."""
        return f"{payload.get('file_path', '')}:{payload.get('start_line', 0)}:{payload.get('end_line', 0)}"

    def _reciprocal_rank_fusion(self, vector_results: List[tuple],
                                  bm25_results: List[tuple],
                                  k: int = 60) -> List[dict]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF is a robust method for combining ranked lists that doesn't require
        score normalization across different retrieval methods.

        Args:
            vector_results: List of (key, score, payload) from vector search
            bm25_results: List of (key, score, payload) from BM25 search
            k: Constant to prevent high ranks from dominating (default: 60)

        Returns:
            List of payloads sorted by fused score
        """
        scores: Dict[str, float] = {}
        payloads: Dict[str, dict] = {}

        # Process vector results
        for rank, (key, _, payload) in enumerate(vector_results, start=1):
            scores[key] = scores.get(key, 0) + self.vector_weight / (k + rank)
            payloads[key] = payload

        # Process BM25 results
        for rank, (key, _, payload) in enumerate(bm25_results, start=1):
            scores[key] = scores.get(key, 0) + self.bm25_weight / (k + rank)
            if key not in payloads:
                payloads[key] = payload

        # Sort by fused score
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        return [payloads[key] for key in sorted_keys]

    def _deduplicate_results(self, results: List[dict], limit: int) -> List[dict]:
        """Deduplicate results by file path, keeping highest-ranked chunk per file."""
        seen_files = set()
        deduplicated = []

        for res in results:
            file_path = res.get("file_path", "")
            if file_path not in seen_files:
                deduplicated.append(res)
                seen_files.add(file_path)
            if len(deduplicated) >= limit:
                break

        return deduplicated
