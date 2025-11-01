"""
Advanced RAG Comparison - Main Package
"""
__version__ = "0.1.0"
__author__ = "Your Name"

from src.config import Config
from src.llm import get_llm, OllamaLLM
from src.embeddings import get_embed_model, get_reranker
from src.utils import load_documents, setup_rag_system

__all__ = [
    "Config",
    "get_llm",
    "OllamaLLM",
    "get_embed_model",
    "get_reranker",
    "load_documents",
    "setup_rag_system",
]