"""Unit tests for vector database implementation."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.vector_db.vector_database import VectorDatabase


class TestVectorDatabase(unittest.TestCase):
    """Test cases for VectorDatabase class."""

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

    def test_init(self):
        """Test database initialization."""
        self.assertEqual(self.db.dimension, 384)
        self.assertEqual(len(self.db.documents), 0)
        self.assertTrue(Path(self.temp_dir).exists())

    def test_add_document(self):
        """Test adding document to database."""
        text = "Test document"
        metadata = {"type": "test"}

        doc_id = self.db.add_document(text, metadata)

        self.assertEqual(doc_id, 0)
        self.assertEqual(len(self.db.documents), 1)
        self.assertEqual(self.db.documents[0]["text"], text)
        self.assertEqual(self.db.documents[0]["metadata"], metadata)

    def test_search(self):
        """Test searching documents."""
        # Add test documents
        docs = [
            ("The quick brown fox", {"type": "animal"}),
            ("Lazy dog sleeping", {"type": "animal"}),
            ("Python programming", {"type": "code"}),
        ]

        for text, metadata in docs:
            self.db.add_document(text, metadata)

        # Search
        results = self.db.search("fox jumping", limit=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "The quick brown fox")
        self.assertTrue("similarity" in results[0])
        self.assertTrue("rank" in results[0])

    def test_search_with_threshold(self):
        """Test search with similarity threshold."""
        self.db.add_document("Specific technical term", {"type": "test"})

        # Search with high threshold
        results = self.db.search("completely different topic", threshold=0.99)
        self.assertEqual(len(results), 0)

    def test_get_document(self):
        """Test getting document by ID."""
        text = "Test document"
        doc_id = self.db.add_document(text)

        document = self.db.get_document(doc_id)

        self.assertIsNotNone(document)
        self.assertEqual(document["text"], text)

    def test_delete_document(self):
        """Test deleting document."""
        doc_id = self.db.add_document("Test document")

        success = self.db.delete_document(doc_id)

        self.assertTrue(success)
        self.assertTrue(self.db.documents[doc_id]["deleted"])

    def test_persistence(self):
        """Test database persistence."""
        # Add document
        self.db.add_document("Test persistence")

        # Create new instance
        new_db = VectorDatabase(self.temp_dir)

        self.assertEqual(len(new_db.documents), 1)
        self.assertEqual(new_db.documents[0]["text"], "Test persistence")
