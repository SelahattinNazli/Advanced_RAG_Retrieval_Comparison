"""
Tests for utility functions
"""
import pytest
import tempfile
from pathlib import Path
from src.utils import (
    get_file_stats,
    format_retrieved_nodes,
    create_sample_eval_questions,
)
from llama_index.core.schema import TextNode, NodeWithScore


def test_get_file_stats_empty_dir():
    """Test file stats on empty directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        stats = get_file_stats(Path(temp_dir))
        assert stats["total_files"] == 0
        assert stats["total_size_mb"] == 0


def test_get_file_stats_with_files():
    """Test file stats with actual files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        (temp_path / "test1.txt").write_text("Hello")
        (temp_path / "test2.txt").write_text("World")
        
        stats = get_file_stats(temp_path)
        assert stats["total_files"] == 2
        assert ".txt" in stats["file_types"]
        assert stats["file_types"][".txt"] == 2


def test_format_retrieved_nodes():
    """Test formatting of retrieved nodes"""
    # Create mock nodes
    node1 = TextNode(text="This is a test node for formatting.")
    node2 = TextNode(text="Another test node with different content.")
    
    nodes = [
        NodeWithScore(node=node1, score=0.95),
        NodeWithScore(node=node2, score=0.85),
    ]
    
    formatted = format_retrieved_nodes(nodes)
    
    assert "Node 1" in formatted
    assert "Node 2" in formatted
    assert "0.9500" in formatted
    assert "0.8500" in formatted


def test_create_sample_eval_questions():
    """Test sample question creation"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "questions.json"
        
        questions = create_sample_eval_questions(output_path)
        
        assert len(questions) > 0
        assert output_path.exists()
        assert all("question" in q for q in questions)
        assert all("id" in q for q in questions)