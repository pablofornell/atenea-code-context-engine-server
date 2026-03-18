import os
from dataclasses import dataclass
from typing import List, Optional
import tree_sitter_languages
from tree_sitter import Node

@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str

class Chunker:
    def __init__(self):
        # We'll support a few common languages initially
        self.supported_langs = {
            "kt": "kotlin",
            "java": "java",
            "xml": "xml",
            "py": "python",
            "gradle": "kotlin",  # Build scripts are often Kotlin
            "kts": "kotlin",
        }

    def chunk_file(self, file_path: str, content: str) -> List[Chunk]:
        ext = file_path.split(".")[-1].lower()
        if ext not in self.supported_langs:
            return self._generic_chunk(file_path, content, "text")
        
        lang_name = self.supported_langs[ext]
        try:
            parser = tree_sitter_languages.get_parser(lang_name)
            tree = parser.parse(bytes(content, "utf8"))
            return self._ast_chunk(file_path, content, tree, lang_name)
        except Exception:
            # Fallback to generic chunking if AST parsing fails
            return self._generic_chunk(file_path, content, lang_name)

    def _ast_chunk(self, file_path: str, content: str, tree, language: str) -> List[Chunk]:
        lines = content.splitlines()
        chunks = []
        
        # We want to identify large structural nodes (classes, functions)
        root_node = tree.root_node
        
        # For simplicity in MVP, we grab the root node or large children
        # In a more advanced version, we'd recursively find nodes of a certain size
        # Augment often returns the most "relevant" chunk. 
        # Here we'll implement a simple strategy: if file is small, one chunk.
        # If large, split by top-level nodes.
        
        if len(lines) <= 50:
            chunks.append(Chunk(file_path, 1, len(lines), content, language))
            return chunks

        # Find top-level structural nodes
        significant_nodes = []
        for child in root_node.children:
            # Heuristic for "significant": classes, functions, or large blocks
            if child.type in ["class_declaration", "function_declaration", "method_declaration", "tag"]:
                significant_nodes.append(child)
        
        if not significant_nodes:
            return self._generic_chunk(file_path, content, language)

        for node in significant_nodes:
            start_row = node.start_point[0]
            end_row = node.end_point[0]
            
            # Extract content for this node
            node_lines = lines[start_row : end_row + 1]
            node_content = "\n".join(node_lines)
            
            chunks.append(Chunk(
                file_path=file_path,
                start_line=start_row + 1,
                end_line=end_row + 1,
                content=node_content,
                language=language
            ))
            
        return chunks

    def _generic_chunk(self, file_path: str, content: str, language: str) -> List[Chunk]:
        # Simple line-based chunking with overlap
        lines = content.splitlines()
        chunk_size = 50
        overlap = 5
        chunks = []
        
        if len(lines) <= chunk_size:
            chunks.append(Chunk(file_path, 1, len(lines), content, language))
            return chunks
            
        for i in range(0, len(lines), chunk_size - overlap):
            end = min(i + chunk_size, len(lines))
            chunk_content = "\n".join(lines[i:end])
            chunks.append(Chunk(
                file_path=file_path,
                start_line=i + 1,
                end_line=end,
                content=chunk_content,
                language=language
            ))
            if end == len(lines):
                break
                
        return chunks
