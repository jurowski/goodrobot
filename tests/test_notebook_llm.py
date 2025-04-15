import os
from unittest.mock import Mock, patch

import pytest

from src.notebook_llm.memory import NotebookMemory
from src.notebook_llm.processor import NotebookProcessor


@pytest.fixture
def notebook_memory():
    return NotebookMemory()


@pytest.fixture
def notebook_processor():
    return NotebookProcessor()


def test_memory_initialization(notebook_memory):
    assert notebook_memory is not None
    assert hasattr(notebook_memory, "vector_dimension")
    assert hasattr(notebook_memory, "similarity_threshold")


def test_memory_add_and_retrieve(notebook_memory):
    test_item = {
        "text": "test note",
        "embedding": [0.1] * notebook_memory.vector_dimension,
        "timestamp": 1234567890,
    }

    notebook_memory.add_item(test_item)
    similar_items = notebook_memory.get_similar_items(
        [0.1] * notebook_memory.vector_dimension
    )

    assert len(similar_items) > 0
    assert similar_items[0]["text"] == "test note"


def test_processor_initialization(notebook_processor):
    assert notebook_processor is not None
    assert hasattr(notebook_processor, "memory")


@patch("openai.Embedding.create")
def test_process_text(mock_embedding, notebook_processor):
    mock_embedding.return_value = {"data": [{"embedding": [0.1] * 768}]}

    result = notebook_processor.process_text("test note")
    assert result is not None
    assert "embedding" in result
