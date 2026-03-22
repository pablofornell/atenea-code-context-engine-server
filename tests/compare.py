import asyncio
import argparse
from typing import List
import pprint
from atenea_server.chunker import Chunker
from atenea_server.embedder import Embedder
from atenea_server.vector_store import VectorStore
from atenea_server.retriever import Retriever
from atenea_server.formatter import Formatter

async def compare_queries(codebase_path: str, queries: List[str]):
    print(f"--- ATENEA VERIFICATION ---")
    print(f"Codebase: {codebase_path}\n")

    chunker = Chunker()
    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever(embedder, vector_store)
    formatter = Formatter()

    for query in queries:
        print(f"\nQUERY: '{query}'")
        print("-" * 50)
        
        # In a real environment, we'd have Augment's output to compare against.
        # Since we're in 'atenea' simulation, we just show what Atenea would return.
        
        chunks = await retriever.retrieve(query, limit=5)
        formatted = formatter.format(chunks)
        
        print(f"Found {len(chunks)} files.")
        if chunks:
            print("First 2 snippets paths:")
            for chunk in chunks[:2]:
                print(f" - {chunk['file_path']} (lines {chunk['start_line']}-{chunk['end_line']})")
        
        print("\n--- SAMPLE FORMATTED OUTPUT (FIRST 500 CHARS) ---")
        print(formatted[:500] + "...")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebase", default="./codebases/Android-SimpleTimeTracker")
    parser.add_argument("--queries", nargs="+", default=["RecordTypeDBO", "Timer logic"])
    
    args = parser.parse_args()
    
    try:
        asyncio.run(compare_queries(args.codebase, args.queries))
    except Exception as e:
        print(f"Error during verification: {e}")
        print("\nNOTE: Ensure Docker services (Qdrant, Ollama) are running and Ollama has 'nomic-embed-text' pulled.")
