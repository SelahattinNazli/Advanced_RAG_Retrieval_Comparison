"""
Configuration management for Advanced RAG Comparison project
"""
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration class"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    STORAGE_DIR = PROJECT_ROOT / "storage"
    
    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120.0"))
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    
    # Reranker Configuration
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "2"))
    
    # Basic RAG Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "6"))
    
    # Sentence Window Configuration
    SENTENCE_WINDOW_SIZE = int(os.getenv("SENTENCE_WINDOW_SIZE", "3"))
    
    # Auto-Merging Configuration
    AUTO_MERGE_CHUNK_SIZES_STR = os.getenv("AUTO_MERGE_CHUNK_SIZES", "2048,512,128")
    AUTO_MERGE_CHUNK_SIZES: List[int] = [
        int(x.strip()) for x in AUTO_MERGE_CHUNK_SIZES_STR.split(",")
    ]
    
    # Evaluation Configuration
    EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "data/eval_questions.json")
    NUM_EVAL_QUESTIONS = int(os.getenv("NUM_EVAL_QUESTIONS", "20"))
    
    # Streamlit Configuration
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_index_dir(cls, index_name: str) -> Path:
        """Get directory path for a specific index"""
        index_dir = cls.STORAGE_DIR / index_name
        index_dir.mkdir(parents=True, exist_ok=True)
        return index_dir
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check if Ollama URL is accessible
        try:
            import requests
            response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                errors.append(f"Ollama not accessible at {cls.OLLAMA_BASE_URL}")
        except Exception as e:
            errors.append(f"Failed to connect to Ollama: {e}")
        
        # Check chunk sizes
        if cls.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE must be positive")
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        # Check auto-merge chunk sizes are in descending order
        if cls.AUTO_MERGE_CHUNK_SIZES != sorted(cls.AUTO_MERGE_CHUNK_SIZES, reverse=True):
            errors.append("AUTO_MERGE_CHUNK_SIZES must be in descending order")
        
        if errors:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("✅ Configuration validated successfully")
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*50)
        print("🔧 Current Configuration")
        print("="*50)
        print(f"\n📁 Paths:")
        print(f"  Project Root: {cls.PROJECT_ROOT}")
        print(f"  Data Dir: {cls.DATA_DIR}")
        print(f"  Storage Dir: {cls.STORAGE_DIR}")
        
        print(f"\n🤖 Ollama:")
        print(f"  Base URL: {cls.OLLAMA_BASE_URL}")
        print(f"  Model: {cls.OLLAMA_MODEL}")
        print(f"  Temperature: {cls.OLLAMA_TEMPERATURE}")
        
        print(f"\n🔤 Embeddings:")
        print(f"  Model: {cls.EMBEDDING_MODEL}")
        print(f"  Device: {cls.EMBEDDING_DEVICE}")
        
        print(f"\n🎯 Retrieval:")
        print(f"  Chunk Size: {cls.CHUNK_SIZE}")
        print(f"  Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"  Top K: {cls.SIMILARITY_TOP_K}")
        print(f"  Rerank Top N: {cls.RERANK_TOP_N}")
        
        print(f"\n🪟 Sentence Window:")
        print(f"  Window Size: {cls.SENTENCE_WINDOW_SIZE}")
        
        print(f"\n🔄 Auto-Merging:")
        print(f"  Chunk Sizes: {cls.AUTO_MERGE_CHUNK_SIZES}")
        
        print("\n" + "="*50 + "\n")


# Create directories on import
Config.create_directories()