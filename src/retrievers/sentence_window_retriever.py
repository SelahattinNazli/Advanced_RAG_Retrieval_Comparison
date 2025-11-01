"""
Sentence Window Retriever - Advanced retrieval with context windows
"""
from typing import List, Optional, Any
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from src.config import Config
from src.utils import save_index, load_index, check_index_exists


class SentenceWindowRetriever:
    """Sentence Window retriever with expandable context"""
    
    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        llm: Any = None,
        embed_model: Any = None,
        reranker: Any = None,
        window_size: Optional[int] = None,
        similarity_top_k: Optional[int] = None,
        index_name: str = "sentence_window_index",
        force_rebuild: bool = False,
    ):
        """
        Initialize Sentence Window Retriever
        
        Args:
            documents: List of documents to index
            llm: Language model
            embed_model: Embedding model
            reranker: Reranker postprocessor
            window_size: Number of sentences before/after to include as context
            similarity_top_k: Number of similar chunks to retrieve
            index_name: Name for saving/loading index
            force_rebuild: Force rebuild even if index exists
        """
        self.index_name = index_name
        self.window_size = window_size or Config.SENTENCE_WINDOW_SIZE
        self.similarity_top_k = similarity_top_k or Config.SIMILARITY_TOP_K
        
        # Set global settings
        if llm:
            Settings.llm = llm
        if embed_model:
            Settings.embed_model = embed_model
        
        self.embed_model = embed_model
        self.reranker = reranker
        
        print(f"\n{'='*60}")
        print(f"🪟 Sentence Window Retriever Configuration")
        print(f"{'='*60}")
        print(f"Window Size: {self.window_size}")
        print(f"Similarity Top K: {self.similarity_top_k}")
        print(f"Index Name: {self.index_name}")
        print(f"{'='*60}\n")
        
        # Build or load index
        if force_rebuild or not check_index_exists(self.index_name):
            if documents is None:
                raise ValueError("Documents required for building new index")
            self.index = self._build_index(documents)
        else:
            print(f"📂 Loading existing index: {self.index_name}")
            self.index = load_index(self.index_name, embed_model, llm)
        
        # Create query engine
        self.query_engine = self._create_query_engine()
    
    def _build_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Build vector store index with sentence window nodes"""
        print(f"🔨 Building Sentence Window index...")
        print(f"   Documents: {len(documents)}")
        
        # Create sentence window node parser
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=self.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        
        # Parse documents into nodes
        nodes = node_parser.get_nodes_from_documents(documents)
        print(f"   Created {len(nodes)} sentence nodes")
        
        # Create index
        index = VectorStoreIndex(nodes, embed_model=self.embed_model)
        
        # Save index
        save_index(index, self.index_name)
        
        print(f"✅ Sentence Window index built successfully")
        return index
    
    def _create_query_engine(self):
        """Create query engine with metadata replacement and optional reranker"""
        node_postprocessors = []
        
        # Add metadata replacement to expand context window
        postproc = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )
        node_postprocessors.append(postproc)
        print(f"✅ Metadata replacement enabled (window expansion)")
        
        # Add reranker if available
        if self.reranker:
            node_postprocessors.append(self.reranker)
            print(f"✅ Reranker enabled")
        
        query_engine = self.index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
            node_postprocessors=node_postprocessors,
        )
        
        return query_engine
    
    def query(self, query_str: str, return_nodes: bool = False):
        """
        Query the index with sentence window context
        
        Args:
            query_str: Query string
            return_nodes: Whether to return retrieved nodes
            
        Returns:
            Response object or tuple of (response, nodes)
        """
        response = self.query_engine.query(query_str)
        
        if return_nodes:
            nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
            return response, nodes
        
        return response
    
    def retrieve(self, query_str: str) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes with window context
        
        Args:
            query_str: Query string
            
        Returns:
            List of retrieved nodes with scores
        """
        retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )
        nodes = retriever.retrieve(query_str)
        
        # Apply metadata replacement to expand windows
        postproc = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )
        nodes = postproc.postprocess_nodes(nodes)
        
        return nodes
    
    def get_retriever_name(self) -> str:
        """Get retriever name for identification"""
        return f"Sentence Window (size={self.window_size})"
    
    def get_config(self) -> dict:
        """Get retriever configuration"""
        return {
            "name": self.get_retriever_name(),
            "window_size": self.window_size,
            "similarity_top_k": self.similarity_top_k,
            "reranker_enabled": self.reranker is not None,
        }


def build_sentence_window_retriever(
    documents: List[Document],
    llm: Any,
    embed_model: Any,
    reranker: Any = None,
    window_size: Optional[int] = None,
    **kwargs
) -> SentenceWindowRetriever:
    """
    Convenience function to build Sentence Window retriever
    
    Args:
        documents: Documents to index
        llm: Language model
        embed_model: Embedding model
        reranker: Reranker postprocessor
        window_size: Sentence window size
        **kwargs: Additional arguments
        
    Returns:
        SentenceWindowRetriever instance
    """
    return SentenceWindowRetriever(
        documents=documents,
        llm=llm,
        embed_model=embed_model,
        reranker=reranker,
        window_size=window_size,
        **kwargs
    )


if __name__ == "__main__":
    # Test sentence window retriever
    from src.utils import load_documents, setup_rag_system
    
    print("🧪 Testing Sentence Window Retriever...")
    
    # Setup system
    system = setup_rag_system()
    
    # Load documents
    documents = load_documents()
    
    if documents:
        # Test different window sizes
        for window_size in [1, 3, 5]:
            print(f"\n{'='*60}")
            print(f"Testing window_size = {window_size}")
            print(f"{'='*60}")
            
            retriever = build_sentence_window_retriever(
                documents=documents,
                llm=system["llm"],
                embed_model=system["embed_model"],
                reranker=system["reranker"],
                window_size=window_size,
                index_name=f"sentence_window_{window_size}",
                force_rebuild=True,
            )
            
            # Test query
            test_query = "What is the main topic?"
            print(f"\n🔍 Test Query: {test_query}")
            response, nodes = retriever.query(test_query, return_nodes=True)
            
            print(f"\n💬 Response:\n{response}\n")
            print(f"📚 Retrieved {len(nodes)} nodes")