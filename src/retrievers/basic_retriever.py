"""
Basic RAG Retriever - Baseline implementation
"""
from typing import List, Optional, Any
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.node_parser import SentenceSplitter
from src.config import Config
from src.utils import save_index, load_index, check_index_exists


class BasicRetriever:
    """Basic RAG retriever with simple chunking"""
    
    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        llm: Any = None,
        embed_model: Any = None,
        reranker: Any = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        similarity_top_k: Optional[int] = None,
        index_name: str = "basic_index",
        force_rebuild: bool = False,
    ):
        """
        Initialize Basic RAG Retriever
        
        Args:
            documents: List of documents to index
            llm: Language model
            embed_model: Embedding model
            reranker: Reranker postprocessor
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            similarity_top_k: Number of similar chunks to retrieve
            index_name: Name for saving/loading index
            force_rebuild: Force rebuild even if index exists
        """
        self.index_name = index_name
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.similarity_top_k = similarity_top_k or Config.SIMILARITY_TOP_K
        
        # Set global settings
        if llm:
            Settings.llm = llm
        if embed_model:
            Settings.embed_model = embed_model
        
        self.embed_model = embed_model
        self.reranker = reranker
        
        print(f"\n{'='*60}")
        print(f"🔧 Basic Retriever Configuration")
        print(f"{'='*60}")
        print(f"Chunk Size: {self.chunk_size}")
        print(f"Chunk Overlap: {self.chunk_overlap}")
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
        """Build vector store index from documents"""
        print(f"🔨 Building Basic RAG index...")
        print(f"   Documents: {len(documents)}")
        
        # Create node parser
        node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        # Parse documents into nodes
        nodes = node_parser.get_nodes_from_documents(documents)
        print(f"   Created {len(nodes)} chunks")
        
        # Create index
        index = VectorStoreIndex(nodes, embed_model=self.embed_model)
        
        # Save index
        save_index(index, self.index_name)
        
        print(f"✅ Basic RAG index built successfully")
        return index
    
    def _create_query_engine(self):
        """Create query engine with optional reranker"""
        node_postprocessors = []
        
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
        Query the index
        
        Args:
            query_str: Query string
            return_nodes: Whether to return retrieved nodes
            
        Returns:
            Response object or tuple of (response, nodes)
        """
        response = self.query_engine.query(query_str)
        
        if return_nodes:
            # Get source nodes
            nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
            return response, nodes
        
        return response
    
    def retrieve(self, query_str: str) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes without generating response
        
        Args:
            query_str: Query string
            
        Returns:
            List of retrieved nodes with scores
        """
        retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )
        nodes = retriever.retrieve(query_str)
        return nodes
    
    def get_retriever_name(self) -> str:
        """Get retriever name for identification"""
        return "Basic RAG"
    
    def get_config(self) -> dict:
        """Get retriever configuration"""
        return {
            "name": self.get_retriever_name(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "similarity_top_k": self.similarity_top_k,
            "reranker_enabled": self.reranker is not None,
        }


def build_basic_retriever(
    documents: List[Document],
    llm: Any,
    embed_model: Any,
    reranker: Any = None,
    **kwargs
) -> BasicRetriever:
    """
    Convenience function to build Basic RAG retriever
    
    Args:
        documents: Documents to index
        llm: Language model
        embed_model: Embedding model
        reranker: Reranker postprocessor
        **kwargs: Additional arguments for BasicRetriever
        
    Returns:
        BasicRetriever instance
    """
    return BasicRetriever(
        documents=documents,
        llm=llm,
        embed_model=embed_model,
        reranker=reranker,
        **kwargs
    )


if __name__ == "__main__":
    # Test basic retriever
    from src.utils import load_documents, setup_rag_system
    
    print("🧪 Testing Basic Retriever...")
    
    # Setup system
    system = setup_rag_system()
    
    # Load documents
    documents = load_documents()
    
    if documents:
        # Build retriever
        retriever = build_basic_retriever(
            documents=documents,
            llm=system["llm"],
            embed_model=system["embed_model"],
            reranker=system["reranker"],
            force_rebuild=True,
        )
        
        # Test query
        test_query = "What is the main topic?"
        print(f"\n🔍 Test Query: {test_query}")
        response, nodes = retriever.query(test_query, return_nodes=True)
        
        print(f"\n💬 Response:\n{response}\n")
        print(f"📚 Retrieved {len(nodes)} nodes")