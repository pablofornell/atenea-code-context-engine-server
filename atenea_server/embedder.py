import httpx
import os
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Default configuration - can be overridden via environment variables
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class Embedder:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.model = model or os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        ollama_url = base_url or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.base_url = f"{ollama_url}/api/embed"
        self._dimension: Optional[int] = None

    async def embed(self, texts: List[str], raise_on_error: bool = True) -> List[List[float]]:
        """
        Generate embeddings for the given texts.

        Args:
            texts: List of text strings to embed
            raise_on_error: If True, raises EmbeddingError on failure.
                          If False, returns empty list on failure.

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails and raise_on_error is True
        """
        if not texts:
            return []

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    self.base_url,
                    json={
                        "model": self.model,
                        "input": texts
                    }
                )
                if response.status_code != 200:
                    error_msg = f"Ollama error {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    if raise_on_error:
                        raise EmbeddingError(error_msg)
                    return []

                data = response.json()
                embeddings = data["embeddings"]

                # Cache dimension for validation
                if embeddings and self._dimension is None:
                    self._dimension = len(embeddings[0])

                return embeddings

            except httpx.TimeoutException as e:
                error_msg = f"Ollama embedding timed out: {e}"
                logger.error(error_msg)
                if raise_on_error:
                    raise EmbeddingError(error_msg) from e
                return []

            except httpx.ConnectError as e:
                error_msg = f"Could not connect to Ollama at {self.base_url}: {e}"
                logger.error(error_msg)
                if raise_on_error:
                    raise EmbeddingError(error_msg) from e
                return []

            except Exception as e:
                error_msg = f"Ollama embedding failed ({type(e).__name__}): {e}"
                logger.error(error_msg)
                if raise_on_error:
                    raise EmbeddingError(error_msg) from e
                return []

    async def embed_with_fallback(self, texts: List[str], max_retries: int = 2) -> Tuple[List[List[float]], List[int]]:
        """
        Embed texts with retry logic. Returns successful embeddings and indices of failed texts.

        Args:
            texts: List of text strings to embed
            max_retries: Number of retry attempts for failed batches

        Returns:
            Tuple of (embeddings, failed_indices) where failed_indices contains
            the indices of texts that could not be embedded after retries
        """
        if not texts:
            return [], []

        embeddings = []
        failed_indices = []

        for attempt in range(max_retries + 1):
            try:
                result = await self.embed(texts, raise_on_error=True)
                return result, []
            except EmbeddingError as e:
                if attempt < max_retries:
                    logger.warning(f"Embedding attempt {attempt + 1} failed, retrying... Error: {e}")
                    continue
                else:
                    logger.error(f"Embedding failed after {max_retries + 1} attempts")
                    failed_indices = list(range(len(texts)))
                    return [], failed_indices

        return embeddings, failed_indices
