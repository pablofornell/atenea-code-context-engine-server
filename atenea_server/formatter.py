from typing import List

class Formatter:
    def __init__(self, cap_bytes: int = 20000):
        self.cap_bytes = cap_bytes

    def format(self, chunks: List[dict]) -> str:
        header = "The following code sections were retrieved:\n"
        output = header
        
        for chunk in chunks:
            snippet = self._format_snippet(chunk)
            
            # Check if adding this snippet exceeds the cap
            if len(output.encode("utf-8")) + len(snippet.encode("utf-8")) > self.cap_bytes:
                # If it's the first snippet and still too large, we might need a partial
                # But usually we just stop here
                break
                
            output += "\n" + snippet
            
        return output

    def _format_snippet(self, chunk: dict) -> str:
        lines = []
        lines.append(f"Path: {chunk['file_path']}")
        
        # Add truncation marker at the top if not starting at line 1
        if chunk["start_line"] > 1:
            lines.append("...")
            
        source_lines = chunk["content"].splitlines()
        for i, line_text in enumerate(source_lines):
            line_no = chunk["start_line"] + i
            # Formatting: 5 spaces for line number + tab + code
            lines.append(f"{line_no:6}\t{line_text}")
            
        # Add truncation marker at the bottom
        lines.append("...")
        
        return "\n".join(lines)
