"""Tests for the BM25Index module."""

import pytest
from atenea_server.bm25_index import BM25Index, BM25Document


class TestBM25Index:
    """Test suite for the BM25Index class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.index = BM25Index()

    def test_add_document(self):
        """Test adding a document to the index."""
        self.index.add_document(
            doc_id="doc1",
            file_path="test.py",
            content="def hello_world(): print('Hello')",
            start_line=1,
            end_line=1,
            language="python"
        )
        
        assert self.index.num_docs == 1
        assert "doc1" in self.index.documents

    def test_search_returns_matching_documents(self):
        """Test that search returns documents matching the query."""
        self.index.add_document(
            doc_id="doc1",
            file_path="auth.py",
            content="def authenticate_user(username, password): pass",
            start_line=1,
            end_line=1,
            language="python"
        )
        self.index.add_document(
            doc_id="doc2",
            file_path="utils.py",
            content="def format_date(date): return str(date)",
            start_line=1,
            end_line=1,
            language="python"
        )
        
        results = self.index.search("authenticate user", limit=10)
        
        assert len(results) >= 1
        # The auth document should be ranked higher
        assert results[0][0] == "doc1"

    def test_search_empty_query(self):
        """Test that empty query returns no results."""
        self.index.add_document(
            doc_id="doc1",
            file_path="test.py",
            content="some content",
            start_line=1,
            end_line=1,
            language="python"
        )
        
        results = self.index.search("", limit=10)
        assert len(results) == 0

    def test_search_no_matches(self):
        """Test search with no matching documents."""
        self.index.add_document(
            doc_id="doc1",
            file_path="test.py",
            content="def hello(): pass",
            start_line=1,
            end_line=1,
            language="python"
        )
        
        results = self.index.search("xyznonexistent", limit=10)
        assert len(results) == 0

    def test_camel_case_tokenization(self):
        """Test that camelCase tokens are properly split."""
        self.index.add_document(
            doc_id="doc1",
            file_path="test.py",
            content="getUserById findAllUsers createNewAccount",
            start_line=1,
            end_line=1,
            language="python"
        )
        
        # Should find by partial camelCase word
        results = self.index.search("user", limit=10)
        assert len(results) >= 1

    def test_snake_case_tokenization(self):
        """Test that snake_case tokens are properly split."""
        self.index.add_document(
            doc_id="doc1",
            file_path="test.py",
            content="get_user_by_id find_all_users create_new_account",
            start_line=1,
            end_line=1,
            language="python"
        )
        
        # Should find by partial snake_case word
        results = self.index.search("user", limit=10)
        assert len(results) >= 1

    def test_clear_index(self):
        """Test clearing the index."""
        self.index.add_document(
            doc_id="doc1",
            file_path="test.py",
            content="content",
            start_line=1,
            end_line=1,
            language="python"
        )
        
        self.index.clear()
        
        assert self.index.num_docs == 0
        assert len(self.index.documents) == 0

    def test_update_existing_document(self):
        """Test updating an existing document."""
        self.index.add_document(
            doc_id="doc1",
            file_path="test.py",
            content="old content",
            start_line=1,
            end_line=1,
            language="python"
        )
        self.index.add_document(
            doc_id="doc1",
            file_path="test.py",
            content="new updated content",
            start_line=1,
            end_line=1,
            language="python"
        )
        
        # Should still have only one document
        assert self.index.num_docs == 1
        # Content should be updated
        assert "updated" in self.index.documents["doc1"].content

    def test_limit_results(self):
        """Test that limit parameter works correctly."""
        for i in range(10):
            self.index.add_document(
                doc_id=f"doc{i}",
                file_path=f"test{i}.py",
                content=f"function test content {i}",
                start_line=1,
                end_line=1,
                language="python"
            )
        
        results = self.index.search("function test", limit=3)
        assert len(results) <= 3

