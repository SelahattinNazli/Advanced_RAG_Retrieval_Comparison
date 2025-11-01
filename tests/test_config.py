"""
Tests for configuration module
"""
import pytest
from pathlib import Path
from src.config import Config


def test_config_paths_exist():
    """Test that config paths are properly set"""
    assert Config.PROJECT_ROOT.exists()
    assert Config.DATA_DIR.exists()
    assert Config.STORAGE_DIR.exists()


def test_config_values():
    """Test that config values are set"""
    assert Config.OLLAMA_MODEL
    assert Config.EMBEDDING_MODEL
    assert Config.CHUNK_SIZE > 0
    assert Config.CHUNK_OVERLAP >= 0
    assert Config.CHUNK_OVERLAP < Config.CHUNK_SIZE


def test_auto_merge_chunk_sizes():
    """Test that auto-merge chunk sizes are valid"""
    assert len(Config.AUTO_MERGE_CHUNK_SIZES) >= 2
    # Should be in descending order
    assert Config.AUTO_MERGE_CHUNK_SIZES == sorted(
        Config.AUTO_MERGE_CHUNK_SIZES, reverse=True
    )


def test_get_index_dir():
    """Test index directory creation"""
    index_dir = Config.get_index_dir("test_index")
    assert index_dir.exists()
    assert index_dir.parent == Config.STORAGE_DIR


def test_ollama_config():
    """Test Ollama configuration"""
    assert "localhost" in Config.OLLAMA_BASE_URL or "127.0.0.1" in Config.OLLAMA_BASE_URL
    assert Config.OLLAMA_TEMPERATURE >= 0
    assert Config.OLLAMA_TEMPERATURE <= 1