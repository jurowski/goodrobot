"""Edge case tests for vector database implementation."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.vector_db.vector_database import VectorDatabase


class TestVectorDatabaseEdgeCases(unittest.TestCase):
    """Test edge cases for VectorDatabase class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db = VectorDatabase(self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        self.db.cleanup()
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)

    def test_empty_text_document(self):
        """Test adding document with empty text."""
        with self.assertRaises(ValueError):
            self.db.add_document("")

    def test_none_text_document(self):
        """Test adding document with None text."""
        with self.assertRaises(ValueError):
            self.db.add_document(None)

    def test_very_long_text(self):
        """Test handling very long text documents."""
        # Create a very long document (100KB)
        long_text = "word " * 20000

        doc_id = self.db.add_document(long_text)
        results = self.db.search("word")

        self.assertEqual(results[0]["id"], doc_id)
        self.assertTrue(len(results) > 0)

    def test_special_characters(self):
        """Test handling text with special characters."""
        special_text = "!@#$%^&*()_+{}|:\"<>?~`-=[]\\;',./"
        doc_id = self.db.add_document(special_text)

        document = self.db.get_document(doc_id)
        self.assertEqual(document["text"], special_text)

    def test_unicode_characters(self):
        """Test handling Unicode characters."""
        unicode_text = "Hello 你好 안녕하세요 こんにちは"
        doc_id = self.db.add_document(unicode_text)

        results = self.db.search("Hello")
        self.assertEqual(results[0]["text"], unicode_text)

    def test_duplicate_documents(self):
        """Test handling duplicate documents."""
        text = "Duplicate text"
        doc_id1 = self.db.add_document(text, {"id": 1})
        doc_id2 = self.db.add_document(text, {"id": 2})

        results = self.db.search(text)
        self.assertEqual(len(results), 2)
        self.assertNotEqual(results[0]["metadata"]["id"], results[1]["metadata"]["id"])

    def test_corrupted_save_file(self):
        """Test handling corrupted save file."""
        # Create corrupted JSON file
        db_file = Path(self.temp_dir) / "vector_db.json"
        with open(db_file, "w") as f:
            f.write("corrupted json{")

        # Should handle gracefully and start fresh
        new_db = VectorDatabase(self.temp_dir)
        self.assertEqual(len(new_db.documents), 0)

    def test_missing_index_file(self):
        """Test handling missing index file."""
        # Add document
        self.db.add_document("Test document")

        # Remove index file
        index_file = Path(self.temp_dir) / "vector_db.index"
        os.remove(index_file)

        # Should handle gracefully and start fresh
        new_db = VectorDatabase(self.temp_dir)
        self.assertEqual(len(new_db.documents), 0)

    def test_invalid_dimension(self):
        """Test handling invalid dimension configuration."""
        with self.assertRaises(ValueError):
            VectorDatabase(self.temp_dir, dimension=0)

    def test_search_empty_query(self):
        """Test search with empty query."""
        with self.assertRaises(ValueError):
            self.db.search("")

    def test_search_none_query(self):
        """Test search with None query."""
        with self.assertRaises(ValueError):
            self.db.search(None)

    def test_invalid_document_id(self):
        """Test operations with invalid document ID."""
        # Test get_document
        self.assertIsNone(self.db.get_document(-1))
        self.assertIsNone(self.db.get_document(9999))

        # Test delete_document
        self.assertFalse(self.db.delete_document(-1))
        self.assertFalse(self.db.delete_document(9999))

    def test_search_with_no_documents(self):
        """Test search with empty database."""
        results = self.db.search("test query")
        self.assertEqual(len(results), 0)

    def test_concurrent_modifications(self):
        """Test handling concurrent modifications."""
        # Simulate concurrent access
        doc_id1 = self.db.add_document("Document 1", commit=False)
        doc_id2 = self.db.add_document("Document 2", commit=False)

        # Both documents should be saved on cleanup
        self.db.cleanup()

        new_db = VectorDatabase(self.temp_dir)
        self.assertEqual(len(new_db.documents), 2)

    def test_large_result_set(self):
        """Test handling large result sets."""
        # Add many documents
        for i in range(1000):
            self.db.add_document(f"Document {i}", commit=False)

        # Test with different limit values
        results1 = self.db.search("Document", limit=10)
        self.assertEqual(len(results1), 10)

        results2 = self.db.search("Document", limit=100)
        self.assertEqual(len(results2), 100)

    def test_metadata_edge_cases(self):
        """Test handling various metadata edge cases."""
        test_cases = [
            ({}, "empty dict"),
            ({"nested": {"data": {"deep": True}}}, "nested dict"),
            ({"list": list(range(100))}, "large list"),
            ({"binary": b"binary data"}, "binary data"),
            (None, "None metadata"),
        ]

        for metadata, case_name in test_cases:
            with self.subTest(case=case_name):
                doc_id = self.db.add_document("Test", metadata)
                document = self.db.get_document(doc_id)
                if metadata is None:
                    self.assertEqual(document["metadata"], {})
                else:
                    self.assertEqual(document["metadata"], metadata)

    @patch("faiss.IndexFlatL2")
    def test_faiss_errors(self, mock_index):
        """Test handling FAISS-related errors."""
        # Mock FAISS error during search
        mock_index.return_value.search.side_effect = RuntimeError("FAISS error")

        with self.assertRaises(RuntimeError):
            self.db.search("test query")

    def test_invalid_save_directory(self):
        """Test handling invalid save directory."""
        # Try to create database in a non-existent path
        invalid_path = "/nonexistent/path/db"

        with self.assertRaises(Exception):
            VectorDatabase(invalid_path)

    def test_document_deletion_consistency(self):
        """Test database consistency after deletions."""
        # Add and delete documents
        doc_id1 = self.db.add_document("Document 1")
        doc_id2 = self.db.add_document("Document 2")

        self.db.delete_document(doc_id1)

        # Add new document
        doc_id3 = self.db.add_document("Document 3")

        # Verify consistency
        self.assertTrue(self.db.documents[doc_id1]["deleted"])
        self.assertFalse("deleted" in self.db.documents[doc_id2])
        self.assertEqual(self.db.documents[doc_id3]["text"], "Document 3")
