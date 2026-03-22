"""
BM25 keyword-based search index for hybrid retrieval.

This module provides a simple in-memory BM25 index that can be used alongside
vector search for improved retrieval precision on exact matches.
"""

import re
import math
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BM25Document:
    """A document in the BM25 index."""
    doc_id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    language: str
    tokens: List[str] = field(default_factory=list)


class BM25Index:
    """
    Simple BM25 index for keyword search.
    
    This is an in-memory index that needs to be rebuilt when data changes.
    For production use, consider using a persistent solution like Elasticsearch.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        
        # Document storage
        self.documents: Dict[str, BM25Document] = {}
        
        # Inverted index: token -> set of doc_ids
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Document frequencies: token -> number of documents containing token
        self.doc_frequencies: Dict[str, int] = defaultdict(int)
        
        # Document lengths (number of tokens)
        self.doc_lengths: Dict[str, int] = {}
        
        # Average document length
        self.avg_doc_length: float = 0.0
        
        # Total number of documents
        self.num_docs: int = 0
        
        # Code-aware tokenization patterns
        self._camel_case_pattern = re.compile(r'(?<!^)(?=[A-Z])')
        self._snake_case_pattern = re.compile(r'_+')
        self._non_alphanum_pattern = re.compile(r'[^a-zA-Z0-9_]')

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with code-aware splitting.
        
        Handles camelCase, snake_case, and extracts meaningful tokens from code.
        """
        tokens = []
        
        # Split on whitespace and punctuation first
        words = self._non_alphanum_pattern.sub(' ', text).split()
        
        for word in words:
            # Skip very short tokens
            if len(word) < 2:
                continue
                
            # Add the original word (lowercased)
            word_lower = word.lower()
            tokens.append(word_lower)
            
            # Split camelCase
            camel_parts = self._camel_case_pattern.split(word)
            if len(camel_parts) > 1:
                for part in camel_parts:
                    if len(part) >= 2:
                        tokens.append(part.lower())
            
            # Split snake_case
            snake_parts = self._snake_case_pattern.split(word)
            if len(snake_parts) > 1:
                for part in snake_parts:
                    if len(part) >= 2:
                        tokens.append(part.lower())
        
        return tokens

    def add_document(self, doc_id: str, file_path: str, content: str,
                     start_line: int, end_line: int, language: str):
        """Add a document to the index."""
        tokens = self._tokenize(content)
        
        doc = BM25Document(
            doc_id=doc_id,
            file_path=file_path,
            content=content,
            start_line=start_line,
            end_line=end_line,
            language=language,
            tokens=tokens
        )
        
        # If document exists, remove old data first
        if doc_id in self.documents:
            self._remove_document(doc_id)
        
        self.documents[doc_id] = doc
        self.doc_lengths[doc_id] = len(tokens)
        
        # Update inverted index and document frequencies
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.inverted_index[token].add(doc_id)
            self.doc_frequencies[token] += 1
        
        # Update statistics
        self.num_docs = len(self.documents)
        self._update_avg_doc_length()

    def _remove_document(self, doc_id: str):
        """Remove a document from the index."""
        if doc_id not in self.documents:
            return
            
        doc = self.documents[doc_id]
        unique_tokens = set(doc.tokens)
        
        for token in unique_tokens:
            self.inverted_index[token].discard(doc_id)
            self.doc_frequencies[token] -= 1
            if self.doc_frequencies[token] <= 0:
                del self.doc_frequencies[token]
                del self.inverted_index[token]
        
        del self.documents[doc_id]
        del self.doc_lengths[doc_id]

    def _update_avg_doc_length(self):
        """Update average document length."""
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        else:
            self.avg_doc_length = 0.0

    def _compute_idf(self, token: str) -> float:
        """Compute inverse document frequency for a token."""
        if token not in self.doc_frequencies:
            return 0.0

        df = self.doc_frequencies[token]
        # IDF with smoothing to avoid negative values
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _compute_bm25_score(self, doc_id: str, query_tokens: List[str]) -> float:
        """Compute BM25 score for a document given query tokens."""
        if doc_id not in self.documents:
            return 0.0

        doc = self.documents[doc_id]
        doc_length = self.doc_lengths[doc_id]

        # Count term frequencies in document
        doc_token_counts = defaultdict(int)
        for token in doc.tokens:
            doc_token_counts[token] += 1

        score = 0.0
        for token in query_tokens:
            if token not in doc_token_counts:
                continue

            tf = doc_token_counts[token]
            idf = self._compute_idf(token)

            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, limit: int = 20,
               collection_filter: Optional[str] = None) -> List[Tuple[str, float, dict]]:
        """
        Search the index with a query.

        Args:
            query: Search query string
            limit: Maximum number of results
            collection_filter: Optional filter (not used in simple implementation)

        Returns:
            List of (doc_id, score, payload) tuples sorted by score descending
        """
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Find candidate documents (must contain at least one query token)
        candidate_docs = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidate_docs.update(self.inverted_index[token])

        if not candidate_docs:
            return []

        # Score candidates
        scored_docs = []
        for doc_id in candidate_docs:
            score = self._compute_bm25_score(doc_id, query_tokens)
            if score > 0:
                doc = self.documents[doc_id]
                payload = {
                    "file_path": doc.file_path,
                    "content": doc.content,
                    "start_line": doc.start_line,
                    "end_line": doc.end_line,
                    "language": doc.language,
                }
                scored_docs.append((doc_id, score, payload))

        # Sort by score descending and limit
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:limit]

    def clear(self):
        """Clear all data from the index."""
        self.documents.clear()
        self.inverted_index.clear()
        self.doc_frequencies.clear()
        self.doc_lengths.clear()
        self.avg_doc_length = 0.0
        self.num_docs = 0

    def build_from_vector_store(self, vector_store, collection_name: Optional[str] = None):
        """
        Build BM25 index from existing vector store data.

        Args:
            vector_store: VectorStore instance to read from
            collection_name: Optional collection name filter
        """
        from hashlib import md5

        target = collection_name or vector_store.default_collection

        try:
            offset = None
            indexed_count = 0

            while True:
                response = vector_store.client.scroll(
                    collection_name=target,
                    limit=100,
                    with_payload=True,
                    offset=offset
                )

                for point in response[0]:
                    payload = point.payload
                    file_path = payload.get("file_path", "")
                    content = payload.get("content", "")
                    start_line = payload.get("start_line", 1)
                    end_line = payload.get("end_line", 1)
                    language = payload.get("language", "text")

                    # Generate same ID as vector store
                    id_input = f"{file_path}:{start_line}:{end_line}"
                    doc_id = md5(id_input.encode()).hexdigest()

                    self.add_document(doc_id, file_path, content, start_line, end_line, language)
                    indexed_count += 1

                offset = response[1]
                if offset is None:
                    break

            logger.info(f"Built BM25 index with {indexed_count} documents from collection '{target}'")

        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")

