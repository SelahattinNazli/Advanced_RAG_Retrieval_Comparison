"""
HuggingFace Embeddings for Advanced RAG Comparison
"""
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Optional
from src.config import Config


class EmbeddingModel:
    """Wrapper for HuggingFace embedding model"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize HuggingFace embedding model
        
        Args:
            model_name: Model name from HuggingFace (default from config)
            device: Device to run on - 'cpu' or 'cuda' (default from config)
            batch_size: Batch size for encoding (default from config)
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.device = device or Config.EMBEDDING_DEVICE
        self.batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
        
        print(f"🔄 Loading embedding model: {self.model_name}")
        print(f"   Device: {self.device}")
        
        # Initialize HuggingFace embedding
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.model_name,
            device=self.device,
        )
        
        print(f"✅ Embedding model loaded successfully")
    
    def get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        return self.embed_model.get_text_embedding(text)
    
    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = [
                self.embed_model.get_text_embedding(text)
                for text in batch
            ]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query (same as text embedding for this model)
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        return self.embed_model.get_query_embedding(query)
    
    def get_embed_model(self):
        """Get the underlying LlamaIndex embedding model"""
        return self.embed_model
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension"""
        # Test with a sample text
        sample_embedding = self.get_text_embedding("test")
        return len(sample_embedding)


class RerankerModel:
    """Wrapper for reranker model"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        top_n: Optional[int] = None,
    ):
        """
        Initialize reranker model
        
        Args:
            model_name: Reranker model name (default from config)
            top_n: Number of top results to return (default from config)
        """
        self.model_name = model_name or Config.RERANKER_MODEL
        self.top_n = top_n or Config.RERANK_TOP_N
        
        print(f"🔄 Loading reranker model: {self.model_name}")
        
        try:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            
            self.reranker = SentenceTransformerRerank(
                top_n=self.top_n,
                model=self.model_name,
            )
            
            print(f"✅ Reranker model loaded successfully")
            print(f"   Top N: {self.top_n}")
            
        except Exception as e:
            print(f"⚠️  Failed to load reranker: {e}")
            print(f"   Continuing without reranking...")
            self.reranker = None
    
    def get_reranker(self):
        """Get the reranker postprocessor"""
        return self.reranker


def get_embed_model() -> HuggingFaceEmbedding:
    """
    Convenience function to get configured embedding model
    
    Returns:
        Configured HuggingFace embedding model
    """
    embedding_wrapper = EmbeddingModel()
    return embedding_wrapper.get_embed_model()


def get_reranker():
    """
    Convenience function to get configured reranker
    
    Returns:
        Configured reranker postprocessor or None
    """
    reranker_wrapper = RerankerModel()
    return reranker_wrapper.get_reranker()


# Test on import (optional)
if __name__ == "__main__":
    print("\n🧪 Testing Embedding Model...")
    
    embed_model = EmbeddingModel()
    
    # Test single embedding
    text = "This is a test sentence for embeddings."
    embedding = embed_model.get_text_embedding(text)
    print(f"\nEmbedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test batch embedding
    texts = [
        "First sentence",
        "Second sentence",
        "Third sentence"
    ]
    embeddings = embed_model.get_text_embedding_batch(texts)
    print(f"\nBatch embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
    
    # Test reranker
    print("\n🧪 Testing Reranker Model...")
    reranker = RerankerModel()