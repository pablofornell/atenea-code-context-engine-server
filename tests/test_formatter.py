"""Tests for the Formatter module."""

import pytest
from atenea_server.formatter import Formatter


class TestFormatter:
    """Test suite for the Formatter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = Formatter()

    def test_format_single_chunk(self):
        """Test formatting a single chunk."""
        chunks = [{
            "file_path": "test.py",
            "start_line": 1,
            "end_line": 3,
            "content": "def hello():\n    print('Hello')\n    return True"
        }]
        
        result = self.formatter.format(chunks)
        
        assert "test.py" in result
        assert "def hello():" in result
        assert "print('Hello')" in result

    def test_format_multiple_chunks(self):
        """Test formatting multiple chunks."""
        chunks = [
            {
                "file_path": "auth.py",
                "start_line": 1,
                "end_line": 2,
                "content": "def login():\n    pass"
            },
            {
                "file_path": "utils.py",
                "start_line": 5,
                "end_line": 6,
                "content": "def helper():\n    pass"
            }
        ]
        
        result = self.formatter.format(chunks)
        
        assert "auth.py" in result
        assert "utils.py" in result
        assert "login" in result
        assert "helper" in result

    def test_format_empty_chunks(self):
        """Test formatting empty chunk list."""
        result = self.formatter.format([])
        
        # Should still have the header
        assert "retrieved" in result.lower()

    def test_format_adds_line_numbers(self):
        """Test that line numbers are added to output."""
        chunks = [{
            "file_path": "test.py",
            "start_line": 10,
            "end_line": 12,
            "content": "line1\nline2\nline3"
        }]
        
        result = self.formatter.format(chunks)
        
        # Should contain line numbers
        assert "10" in result
        assert "11" in result
        assert "12" in result

    def test_format_adds_truncation_markers(self):
        """Test that truncation markers are added."""
        chunks = [{
            "file_path": "test.py",
            "start_line": 50,  # Not starting at line 1
            "end_line": 52,
            "content": "some code"
        }]
        
        result = self.formatter.format(chunks)
        
        # Should have truncation marker since not starting at line 1
        assert "..." in result

    def test_format_respects_byte_cap(self):
        """Test that output respects the byte cap."""
        # Create formatter with small cap
        formatter = Formatter(cap_bytes=200)
        
        # Create chunks that would exceed the cap
        chunks = [
            {
                "file_path": f"file{i}.py",
                "start_line": 1,
                "end_line": 10,
                "content": "x" * 100
            }
            for i in range(10)
        ]
        
        result = formatter.format(chunks)
        
        # Result should be within cap (approximately)
        assert len(result.encode("utf-8")) <= 300  # Some buffer for header

    def test_format_preserves_code_content(self):
        """Test that code content is preserved exactly."""
        code = '''def complex_function(a, b, c):
    result = a + b
    if result > c:
        return result * 2
    return result'''
        
        chunks = [{
            "file_path": "test.py",
            "start_line": 1,
            "end_line": 5,
            "content": code
        }]
        
        result = self.formatter.format(chunks)
        
        # All code lines should be present
        assert "complex_function" in result
        assert "result = a + b" in result
        assert "return result * 2" in result

