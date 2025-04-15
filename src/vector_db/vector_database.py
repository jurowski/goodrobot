"""Vector database implementation using FAISS for semantic search."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorDatabase:
    """FAISS-based vector database for semantic search."""

    def __init__(self, data_dir: str, dimension: int = 384):
        """Initialize vector database.

        Args:
            data_dir: Directory for storing database files
            dimension: Dimension of vector embeddings
        """
        self.data_dir = Path(data_dir)
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents: List[Dict[str, Any]] = []

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing database if available
        self._load_database()
        logger.info("Vector database initialized with dimension %d", dimension)

    def _load_database(self) -> None:
        """Load existing database from disk."""
        db_file = self.data_dir / "vector_db.json"
        index_file = self.data_dir / "vector_db.index"

        try:
            if db_file.exists() and index_file.exists():
                # Load document metadata
                with open(db_file, "r") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])

                # Load FAISS index
                self.index = faiss.read_index(str(index_file))
                logger.info(
                    "Loaded existing database with %d documents", len(self.documents)
                )
            else:
                logger.info("No existing database found, starting fresh")

        except Exception as e:
            logger.error("Failed to load database: %s", str(e))
            logger.warning("Starting with fresh database")
            self.documents = []
            self.index = faiss.IndexFlatL2(self.dimension)

    def _save_database(self) -> None:
        """Save database to disk."""
        try:
            # Save document metadata
            db_file = self.data_dir / "vector_db.json"
            data = {
                "documents": self.documents,
                "dimension": self.dimension,
                "updated_at": datetime.now().isoformat(),
            }
            with open(db_file, "w") as f:
                json.dump(data, f, indent=2)

            # Save FAISS index
            index_file = self.data_dir / "vector_db.index"
            faiss.write_index(self.index, str(index_file))

            logger.info("Database saved successfully")

        except Exception as e:
            logger.error("Failed to save database: %s", str(e))
            raise

    def add_document(
        self, text: str, metadata: Optional[Dict[str, Any]] = None, commit: bool = True
    ) -> int:
        """Add document to database.

        Args:
            text: Document text
            metadata: Optional document metadata
            commit: Whether to commit changes to disk

        Returns:
            Document ID

        Raises:
            ValueError: If text is invalid
            RuntimeError: If embedding fails
        """
        if not text:
            raise ValueError("Document text cannot be empty")

        try:
            # Create embedding
            embedding = self.model.encode([text])[0]
            if len(embedding) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: got {len(embedding)}, "
                    f"expected {self.dimension}"
                )

            # Add to FAISS index
            self.index.add(np.array([embedding]))

            # Store document
            doc_id = len(self.documents)
            document = {
                "id": doc_id,
                "text": text,
                "metadata": metadata or {},
                "added_at": datetime.now().isoformat(),
            }
            self.documents.append(document)

            # Save if requested
            if commit:
                self._save_database()

            logger.info("Added document %d", doc_id)
            return doc_id

        except Exception as e:
            logger.error("Failed to add document: %s", str(e))
            raise

    def search(
        self,
        query: str,
        limit: int = 5,
        threshold: Optional[float] = None,
        return_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Optional similarity threshold
            return_scores: Whether to include similarity scores

        Returns:
            List of matching documents

        Raises:
            ValueError: If query is invalid
            RuntimeError: If search fails
        """
        if not query:
            raise ValueError("Search query cannot be empty")
        if limit < 1:
            raise ValueError("Limit must be positive")

        try:
            # Create query embedding
            query_embedding = self.model.encode([query])[0]

            # Search index
            distances, indices = self.index.search(np.array([query_embedding]), limit)

            # Format results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx >= len(self.documents):
                    continue

                similarity = 1 / (1 + distance)
                if threshold and similarity < threshold:
                    continue

                result = dict(self.documents[idx])
                if return_scores:
                    result.update({"similarity": similarity, "rank": i + 1})
                results.append(result)

            logger.info("Search returned %d results", len(results))
            return results

        except Exception as e:
            logger.error("Search failed: %s", str(e))
            raise

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        if 0 <= doc_id < len(self.documents):
            return dict(self.documents[doc_id])
        return None

    def delete_document(self, doc_id: int, commit: bool = True) -> bool:
        """Delete document from database.

        Args:
            doc_id: Document ID
            commit: Whether to commit changes to disk

        Returns:
            True if document was deleted, False otherwise
        """
        if 0 <= doc_id < len(self.documents):
            # Mark document as deleted
            self.documents[doc_id]["deleted"] = True
            self.documents[doc_id]["deleted_at"] = datetime.now().isoformat()

            # Save if requested
            if commit:
                self._save_database()

            logger.info("Deleted document %d", doc_id)
            return True

        logger.warning("Document %d not found", doc_id)
        return False

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._save_database()
            logger.info("Database cleanup completed")
        except Exception as e:
            logger.error("Cleanup failed: %s", str(e))
