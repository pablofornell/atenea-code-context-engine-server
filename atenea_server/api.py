import asyncio
import logging
import os
from aiohttp import web
from typing import List

from .chunker import Chunker, Chunk
from .embedder import Embedder
from .vector_store import VectorStore
from .indexer import Indexer
from .retriever import Retriever
from .formatter import Formatter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("atenea.api")

class AteneaAPI:
    def __init__(self):
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.indexer = Indexer(self.chunker, self.embedder, self.vector_store)
        self.retriever = Retriever(self.embedder, self.vector_store)
        self.formatter = Formatter()

    async def handle_status(self, request):
        has_data = self.vector_store.has_data()
        return web.json_response({
            "status": "ok",
            "indexed": has_data,
            "engine": "Atenea Context Engine"
        })

    async def handle_index(self, request):
        try:
            data = await request.json()
            files = data.get("files", [])
            
            if not files:
                return web.json_response({"error": "No files provided"}, status=400)

            all_chunks = []
            for f in files:
                path = f.get("path")
                content = f.get("content", "")
                if not path or not content.strip():
                    continue
                
                # Re-use chunker
                file_chunks = self.chunker.chunk_file(path, content)
                all_chunks.extend(file_chunks)

            if not all_chunks:
                return web.json_response({"status": "ok", "message": "No chunks found to index", "chunks": 0})

            # Re-use indexer logic for embedding and storing
            # This is a bit of a hack since indexer doesn't have a public "index_chunks" method yet
            # but we can implement it here or refactor indexer.
            
            batch_size = 50
            semaphore = asyncio.Semaphore(2)

            async def process_batch(batch_idx: int, batch_chunks: List[Chunk]):
                async with semaphore:
                    contents = [c.content for c in batch_chunks]
                    embeddings = await self.embedder.embed(contents)
                    self.vector_store.upsert_chunks(batch_chunks, embeddings)
                    logger.info(f"Indexed batch {batch_idx + 1} via API...")

            batches = [all_chunks[i:i+batch_size] for i in range(0, len(all_chunks), batch_size)]
            tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
            await asyncio.gather(*tasks)

            return web.json_response({
                "status": "ok",
                "message": f"Successfully indexed {len(files)} files",
                "chunks": len(all_chunks)
            })
        except Exception as e:
            logger.error(f"Error indexing: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_query(self, request):
        try:
            data = await request.json()
            query = data.get("query")
            limit = data.get("limit", 20)

            if not query:
                return web.json_response({"error": "No query provided"}, status=400)

            chunks = await self.retriever.retrieve(query, limit=limit)
            formatted = self.formatter.format(chunks)

            return web.json_response({
                "status": "ok",
                "results": formatted,
                "count": len(chunks)
            })
        except Exception as e:
            logger.error(f"Error querying: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_clean(self, request):
        try:
            self.vector_store.clear_collection()
            return web.json_response({"status": "ok", "message": "Index cleared"})
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return web.json_response({"error": str(e)}, status=500)

def main():
    api = AteneaAPI()
    app = web.Application(client_max_size=100 * 1024 * 1024)  # 100MB
    app.add_routes([
        web.get('/api/status', api.handle_status),
        web.post('/api/index', api.handle_index),
        web.post('/api/query', api.handle_query),
        web.delete('/api/index', api.handle_clean),
    ])
    
    port = int(os.environ.get("PORT", 8080))
    web.run_app(app, port=port)

if __name__ == "__main__":
    main()
