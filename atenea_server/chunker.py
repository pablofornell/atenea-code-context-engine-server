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
            "js": "javascript",
            "jsx": "javascript",
            "ts": "typescript",
            "tsx": "tsx",
            "go": "go",
            "rs": "rust",
            "rb": "ruby",
            "c": "c",
            "cpp": "cpp",
            "h": "c",
            "hpp": "cpp",
            "cs": "c_sharp",
            "swift": "swift",
            "php": "php",
        }

        # Node types that represent significant code structures (language-agnostic)
        self.significant_node_types = {
            # Classes and similar structures
            "class_definition", "class_declaration", "class_specifier",
            "interface_declaration", "struct_specifier", "enum_declaration",
            "trait_declaration", "impl_item", "module_definition",
            # Functions and methods
            "function_definition", "function_declaration", "method_definition",
            "method_declaration", "function_item", "arrow_function",
            "lambda", "lambda_expression",
            # Decorators (Python) - we want to include these with their target
            "decorated_definition",
            # Other significant blocks
            "export_statement", "const_declaration", "let_declaration",
            "variable_declaration",  # For top-level consts/vars
            "type_alias_declaration", "interface_declaration",
            # XML/HTML
            "tag", "element",
        }

        # Minimum lines for a chunk to be worthwhile on its own
        self.min_chunk_lines = 3
        # Maximum lines before we try to split further
        self.max_chunk_lines = 150

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

        # For very small files, return as single chunk
        if len(lines) <= 30:
            return [Chunk(file_path, 1, len(lines), content, language)]

        # Recursively find all significant nodes
        significant_nodes = self._find_significant_nodes(tree.root_node, lines)

        if not significant_nodes:
            return self._generic_chunk(file_path, content, language)

        chunks = []
        covered_lines = set()

        for node in significant_nodes:
            start_row = node.start_point[0]
            end_row = node.end_point[0]

            # Skip if this range is already covered by a parent chunk
            node_range = set(range(start_row, end_row + 1))
            if node_range.issubset(covered_lines):
                continue

            # Extract content for this node
            node_lines = lines[start_row:end_row + 1]
            node_content = "\n".join(node_lines)

            # If chunk is too large, try to split it into smaller pieces
            if len(node_lines) > self.max_chunk_lines:
                sub_chunks = self._split_large_node(file_path, node, lines, language)
                chunks.extend(sub_chunks)
                for sc in sub_chunks:
                    covered_lines.update(range(sc.start_line - 1, sc.end_line))
            else:
                chunks.append(Chunk(
                    file_path=file_path,
                    start_line=start_row + 1,
                    end_line=end_row + 1,
                    content=node_content,
                    language=language
                ))
                covered_lines.update(node_range)

        # Add any uncovered significant sections (imports, module-level code)
        chunks.extend(self._capture_uncovered_sections(file_path, lines, covered_lines, language))

        # Sort chunks by start line
        chunks.sort(key=lambda c: c.start_line)

        return chunks if chunks else self._generic_chunk(file_path, content, language)

    def _find_significant_nodes(self, node: Node, lines: List[str], depth: int = 0) -> List[Node]:
        """Recursively find all significant code structure nodes."""
        significant = []

        # Check if current node is significant
        if node.type in self.significant_node_types:
            node_lines = node.end_point[0] - node.start_point[0] + 1
            # Only include if it's meaningful (not too small)
            if node_lines >= self.min_chunk_lines:
                significant.append(node)
                # Don't recurse into children of small-medium nodes
                if node_lines <= self.max_chunk_lines:
                    return significant

        # Recurse into children
        for child in node.children:
            significant.extend(self._find_significant_nodes(child, lines, depth + 1))

        return significant

    def _split_large_node(self, file_path: str, node: Node, lines: List[str], language: str) -> List[Chunk]:
        """Split a large node into smaller chunks based on its children."""
        chunks = []

        # Find significant children within this node
        child_nodes = []
        for child in node.children:
            if child.type in self.significant_node_types:
                child_lines = child.end_point[0] - child.start_point[0] + 1
                if child_lines >= self.min_chunk_lines:
                    child_nodes.append(child)

        if child_nodes:
            # Create chunks for each significant child
            for child in child_nodes:
                start_row = child.start_point[0]
                end_row = child.end_point[0]
                node_lines = lines[start_row:end_row + 1]
                node_content = "\n".join(node_lines)

                chunks.append(Chunk(
                    file_path=file_path,
                    start_line=start_row + 1,
                    end_line=end_row + 1,
                    content=node_content,
                    language=language
                ))
        else:
            # No significant children, use generic chunking for this range
            start_row = node.start_point[0]
            end_row = node.end_point[0]
            node_content = "\n".join(lines[start_row:end_row + 1])
            chunks.extend(self._generic_chunk_content(
                file_path, node_content, language,
                start_offset=start_row
            ))

        return chunks

    def _capture_uncovered_sections(self, file_path: str, lines: List[str],
                                     covered_lines: set, language: str) -> List[Chunk]:
        """Capture important uncovered sections like imports and module-level code."""
        chunks = []
        uncovered_start = None

        for i, line in enumerate(lines):
            if i not in covered_lines:
                if uncovered_start is None:
                    uncovered_start = i
            else:
                if uncovered_start is not None:
                    # We have an uncovered section
                    section_lines = lines[uncovered_start:i]
                    # Only create chunk if section has meaningful content
                    if self._has_meaningful_content(section_lines):
                        chunks.append(Chunk(
                            file_path=file_path,
                            start_line=uncovered_start + 1,
                            end_line=i,
                            content="\n".join(section_lines),
                            language=language
                        ))
                    uncovered_start = None

        # Handle trailing uncovered section
        if uncovered_start is not None:
            section_lines = lines[uncovered_start:]
            if self._has_meaningful_content(section_lines):
                chunks.append(Chunk(
                    file_path=file_path,
                    start_line=uncovered_start + 1,
                    end_line=len(lines),
                    content="\n".join(section_lines),
                    language=language
                ))

        return chunks

    def _has_meaningful_content(self, lines: List[str]) -> bool:
        """Check if lines contain meaningful code (not just whitespace/comments)."""
        non_empty = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]
        return len(non_empty) >= 2

    def _generic_chunk_content(self, file_path: str, content: str, language: str,
                                start_offset: int = 0) -> List[Chunk]:
        """Generic chunking for a specific content section."""
        lines = content.splitlines()
        chunk_size = 50
        overlap = 5
        chunks = []

        if len(lines) <= chunk_size:
            return [Chunk(file_path, start_offset + 1, start_offset + len(lines), content, language)]

        for i in range(0, len(lines), chunk_size - overlap):
            end = min(i + chunk_size, len(lines))
            chunk_content = "\n".join(lines[i:end])
            chunks.append(Chunk(
                file_path=file_path,
                start_line=start_offset + i + 1,
                end_line=start_offset + end,
                content=chunk_content,
                language=language
            ))
            if end == len(lines):
                break

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
