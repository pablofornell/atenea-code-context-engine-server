import httpx
from typing import List
import logging

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = f"{base_url}/api/embed"

    async def embed(self, texts: List[str]) -> List[List[float]]:
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
                    logger.error(f"Ollama error {response.status_code}: {response.text}")
                    return [[0.0] * 768 for _ in texts]
                
                data = response.json()
                return data["embeddings"]
            except Exception as e:
                logger.error(f"Ollama embedding failed (type: {type(e).__name__}): {e}")
                return [[0.0] * 768 for _ in texts]
