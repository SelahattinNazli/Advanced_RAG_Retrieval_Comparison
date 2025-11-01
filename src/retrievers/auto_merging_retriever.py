"""
Auto-Merging Retriever - Hierarchical retrieval with automatic merging
"""
from typing import List, Optional, Any
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever as LlamaAutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from src.config import Config
from src.utils import save_index, load_index, check_index_exists


class AutoMergingRetriever:
    """Auto-Merging retriever with hierarchical chunking"""
    
    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        llm: Any = None,
        embed_model: Any = None,
        reranker: Any = None,
        chunk_sizes: Optional[List[int]] = None,
        similarity_top_k: Optional[int] = None,
        index_name: str = "auto_merging_index",
        force_rebuild: bool = False,
    ):
        """
        Initialize Auto-Merging Retriever
        
        Args:
            documents: List of documents to index
            llm: Language model
            embed_model: Embedding model
            reranker: Reranker postprocessor
            chunk_sizes: List of chunk sizes for hierarchy (descending order)
            similarity_top_k: Number of similar chunks to retrieve
            index_name: Name for saving/loading index
            force_rebuild: Force rebuild even if index exists
        """
        self.index_name = index_name
        self.chunk_sizes = chunk_sizes or Config.AUTO_MERGE_CHUNK_SIZES
        self.similarity_top_k = similarity_top_k or Config.SIMILARITY_TOP_K
        
        # Validate chunk sizes are in descending order
        if self.chunk_sizes != sorted(self.chunk_sizes, reverse=True):
            raise ValueError("chunk_sizes must be in descending order")
        
        # Set global settings
        if llm:
            Settings.llm = llm
        if embed_model:
            Settings.embed_model = embed_model
        
        self.embed_model = embed_model
        self.reranker = reranker
        self.storage_context = None
        
        print(f"\n{'='*60}")
        print(f"🔄 Auto-Merging Retriever Configuration")
        print(f"{'='*60}")
        print(f"Chunk Sizes (hierarchical): {self.chunk_sizes}")
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
            self.index, self.storage_context = self._load_index_with_context(llm, embed_model)
        
        # Create query engine
        self.query_engine = self._create_query_engine()
    
    def _build_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Build hierarchical index with auto-merging capability"""
        print(f"🔨 Building Auto-Merging index...")
        print(f"   Documents: {len(documents)}")
        
        # Create hierarchical node parser
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes,
        )
        
        # Parse documents into hierarchical nodes
        nodes = node_parser.get_nodes_from_documents(documents)
        print(f"   Created {len(nodes)} hierarchical nodes")
        
        # Get leaf nodes for indexing
        leaf_nodes = get_leaf_nodes(nodes)
        print(f"   Leaf nodes (smallest chunks): {len(leaf_nodes)}")
        
        # Create storage context to store all nodes
        self.storage_context = StorageContext.from_defaults()
        self.storage_context.docstore.add_documents(nodes)
        
        # Create index using only leaf nodes
        index = VectorStoreIndex(
            leaf_nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
        )
        
        # Save index
        save_index(index, self.index_name)
        
        print(f"✅ Auto-Merging index built successfully")
        return index
    
    def _load_index_with_context(self, llm, embed_model):
        """Load index and storage context"""
        from llama_index.core import load_index_from_storage
        
        index_dir = Config.get_index_dir(self.index_name)
        storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
        
        # Set settings
        Settings.embed_model = embed_model
        if llm:
            Settings.llm = llm
        
        index = load_index_from_storage(storage_context)
        
        return index, storage_context
    
    def _create_query_engine(self):
        """Create query engine with auto-merging retriever"""
        # Create base retriever
        base_retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )
        
        # Wrap with auto-merging retriever
        retriever = LlamaAutoMergingRetriever(
            base_retriever,
            self.storage_context,
            verbose=True,
        )
        print(f"✅ Auto-merging enabled")
        
        # Create node postprocessors
        node_postprocessors = []
        if self.reranker:
            node_postprocessors.append(self.reranker)
            print(f"✅ Reranker enabled")
        
        # Create query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever,
            node_postprocessors=node_postprocessors,
        )
        
        return query_engine
    
    def query(self, query_str: str, return_nodes: bool = False):
        """
        Query the index with auto-merging
        
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
        Retrieve relevant nodes with auto-merging
        
        Args:
            query_str: Query string
            
        Returns:
            List of retrieved nodes with scores (potentially merged)
        """
        base_retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )
        
        retriever = LlamaAutoMergingRetriever(
            base_retriever,
            self.storage_context,
            verbose=True,
        )
        
        nodes = retriever.retrieve(query_str)
        return nodes
    
    def get_retriever_name(self) -> str:
        """Get retriever name for identification"""
        return f"Auto-Merging (sizes={self.chunk_sizes})"
    
    def get_config(self) -> dict:
        """Get retriever configuration"""
        return {
            "name": self.get_retriever_name(),
            "chunk_sizes": self.chunk_sizes,
            "similarity_top_k": self.similarity_top_k,
            "reranker_enabled": self.reranker is not None,
        }


def build_auto_merging_retriever(
    documents: List[Document],
    llm: Any,
    embed_model: Any,
    reranker: Any = None,
    chunk_sizes: Optional[List[int]] = None,
    **kwargs
) -> AutoMergingRetriever:
    """
    Convenience function to build Auto-Merging retriever
    
    Args:
        documents: Documents to index
        llm: Language model
        embed_model: Embedding model
        reranker: Reranker postprocessor
        chunk_sizes: Hierarchical chunk sizes
        **kwargs: Additional arguments
        
    Returns:
        AutoMergingRetriever instance
    """
    return AutoMergingRetriever(
        documents=documents,
        llm=llm,
        embed_model=embed_model,
        reranker=reranker,
        chunk_sizes=chunk_sizes,
        **kwargs
    )


if __name__ == "__main__":
    # Test auto-merging retriever
    from src.utils import load_documents, setup_rag_system
    
    print("🧪 Testing Auto-Merging Retriever...")
    
    # Setup system
    system = setup_rag_system()
    
    # Load documents
    documents = load_documents()
    
    if documents:
        # Test different chunk size configurations
        configs = [
            [2048, 512, 128],
            [1024, 256, 64],
        ]
        
        for chunk_sizes in configs:
            print(f"\n{'='*60}")
            print(f"Testing chunk_sizes = {chunk_sizes}")
            print(f"{'='*60}")
            
            retriever = build_auto_merging_retriever(
                documents=documents,
                llm=system["llm"],
                embed_model=system["embed_model"],
                reranker=system["reranker"],
                chunk_sizes=chunk_sizes,
                index_name=f"auto_merging_{'_'.join(map(str, chunk_sizes))}",
                force_rebuild=True,
            )
            
            # Test query
            test_query = "What is the main topic?"
            print(f"\n🔍 Test Query: {test_query}")
            response, nodes = retriever.query(test_query, return_nodes=True)
            
            print(f"\n💬 Response:\n{response}\n")
            print(f"📚 Retrieved {len(nodes)} nodes (after merging)")