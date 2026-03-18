import os
import asyncio
from typing import List, Set
import logging
from .chunker import Chunker, Chunk
from .embedder import Embedder
from .vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Indexer:
    def __init__(self, chunker: Chunker, embedder: Embedder, vector_store: VectorStore):
        self.chunker = chunker
        self.embeder = embedder
        self.vector_store = vector_store
        self.ignored_dirs = {
            ".git", "build", "node_modules", ".gradle", ".venv", "venv", 
            ".idea", "bin", "obj", "out", "metadata", ".next", "dist", 
            "target", "__pycache__", ".vscode", ".pytest_cache", ".mypy_cache"
        }
        self.binary_exts = {
            ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", 
            ".exe", ".dll", ".so", ".bin", ".jar", ".class", ".aar", ".xcf",
            ".svg", ".ttf", ".otf", ".woff", ".woff2", ".7z", ".tar", ".gz",
            ".dmg", ".iso", ".sqlite"
        }
        self.ignored_files = {
            "gradlew", "gradlew.bat", 
            ".gitignore", "gradle.properties", "settings.gradle", "package-lock.json",
            "yarn.lock", "pnpm-lock.yaml", ".DS_Store"
        }

    async def index_directory(self, root_path: str):
        logger.info(f"Indexing directory: {root_path}")
        
        all_chunks = []
        for root, dirs, files in os.walk(root_path):
            # Prune ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]
            
            for file in files:
                if file in self.ignored_files:
                    continue
                    
                ext = os.path.splitext(file)[1].lower()
                if ext in self.binary_exts:
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_path)
                
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if not content.strip():
                            continue
                            
                        file_chunks = self.chunker.chunk_file(rel_path, content)
                        all_chunks.extend(file_chunks)
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")

        if not all_chunks:
            logger.info("No chunks to index.")
            return

        logger.info(f"Found {len(all_chunks)} chunks. Generating embeddings...")
        
        # Batch processing with parallelism
        batch_size = 50
        semaphore = asyncio.Semaphore(2) # Limit concurrency to avoid overloading Ollama

        async def process_batch(batch_idx: int, batch_chunks: List[Chunk]):
            async with semaphore:
                contents = [c.content for c in batch_chunks]
                embeddings = await self.embeder.embed(contents)
                self.vector_store.upsert_chunks(batch_chunks, embeddings)
                logger.info(f"Indexed batch {batch_idx + 1}...")

        batches = [all_chunks[i:i+batch_size] for i in range(0, len(all_chunks), batch_size)]
        tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
        
        await asyncio.gather(*tasks)
            
        logger.info("Indexing complete.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m atenea.indexer <directory_path>")
        sys.exit(1)
        
    dir_to_index = sys.argv[1]
    chunker = Chunker()
    embedder = Embedder()
    vector_store = VectorStore()
    indexer = Indexer(chunker, embedder, vector_store)
    
    asyncio.run(indexer.index_directory(dir_to_index))
