"""
Tests for retriever modules
Note: These tests require actual embedding models and are skipped by default.
Run with: pytest tests/test_retrievers.py --run-integration
"""
import pytest
from llama_index.core import Document


pytestmark = pytest.mark.skip(reason="Requires embedding models - integration test")
from src.retrievers import (
    BasicRetriever,
    SentenceWindowRetriever,
    AutoMergingRetriever,
)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        Document(text="This is the first test document about machine learning."),
        Document(text="This is the second test document about natural language processing."),
        Document(text="The third document discusses retrieval augmented generation systems."),
    ]


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    # Return None - retrievers will use default
    return None


@pytest.fixture
def mock_embed_model():
    """Mock embedding model for testing"""
    # Return None - retrievers will use default
    return None


def test_basic_retriever_initialization(sample_documents, mock_llm, mock_embed_model):
    """Test basic retriever can be initialized"""
    retriever = BasicRetriever(
        documents=sample_documents,
        llm=mock_llm,
        embed_model=mock_embed_model,
        index_name="test_basic",
        force_rebuild=True,
    )
    
    assert retriever is not None
    assert retriever.get_retriever_name() == "Basic RAG"


def test_sentence_window_retriever_initialization(sample_documents, mock_llm, mock_embed_model):
    """Test sentence window retriever can be initialized"""
    retriever = SentenceWindowRetriever(
        documents=sample_documents,
        llm=mock_llm,
        embed_model=mock_embed_model,
        window_size=3,
        index_name="test_window",
        force_rebuild=True,
    )
    
    assert retriever is not None
    assert "Sentence Window" in retriever.get_retriever_name()


def test_auto_merging_retriever_initialization(sample_documents, mock_llm, mock_embed_model):
    """Test auto-merging retriever can be initialized"""
    retriever = AutoMergingRetriever(
        documents=sample_documents,
        llm=mock_llm,
        embed_model=mock_embed_model,
        chunk_sizes=[512, 128, 32],
        index_name="test_auto",
        force_rebuild=True,
    )
    
    assert retriever is not None
    assert "Auto-Merging" in retriever.get_retriever_name()


def test_retriever_config(sample_documents, mock_llm, mock_embed_model):
    """Test retriever configuration is properly stored"""
    retriever = BasicRetriever(
        documents=sample_documents,
        llm=mock_llm,
        embed_model=mock_embed_model,
        chunk_size=256,
        chunk_overlap=20,
        index_name="test_config",
        force_rebuild=True,
    )
    
    config = retriever.get_config()
    
    assert config["chunk_size"] == 256
    assert config["chunk_overlap"] == 20
    assert "name" in config


def test_invalid_chunk_sizes():
    """Test that invalid chunk sizes raise error"""
    from llama_index.core import Document
    
    with pytest.raises(ValueError):
        # Chunk sizes must be descending
        AutoMergingRetriever(
            documents=[Document(text="test")],
            chunk_sizes=[128, 512, 32],  # Not descending!
            index_name="test_invalid",
            force_rebuild=True,
        )