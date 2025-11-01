"""
Utility functions for Advanced RAG Comparison
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.core.schema import NodeWithScore
from src.config import Config


def load_documents(
    data_dir: Optional[Path] = None,
    required_exts: Optional[List[str]] = None,
    recursive: bool = True,
) -> List[Document]:
    """
    Load documents from directory
    
    Args:
        data_dir: Directory containing documents (default: Config.DATA_DIR)
        required_exts: List of file extensions to load (e.g., ['.pdf', '.txt'])
        recursive: Whether to search recursively
        
    Returns:
        List of Document objects
    """
    data_dir = data_dir or Config.DATA_DIR
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"📂 Loading documents from: {data_dir}")
    
    try:
        reader = SimpleDirectoryReader(
            input_dir=str(data_dir),
            required_exts=required_exts,
            recursive=recursive,
        )
        documents = reader.load_data()
        
        print(f"✅ Loaded {len(documents)} documents")
        
        # Print document info
        for i, doc in enumerate(documents[:5]):  # Show first 5
            filename = doc.metadata.get("file_name", "Unknown")
            print(f"   {i+1}. {filename} ({len(doc.text)} chars)")
        
        if len(documents) > 5:
            print(f"   ... and {len(documents) - 5} more")
        
        return documents
        
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        raise


def save_index(index: Any, index_name: str) -> Path:
    """
    Save index to disk
    
    Args:
        index: LlamaIndex index object
        index_name: Name for the index directory
        
    Returns:
        Path to saved index
    """
    index_dir = Config.get_index_dir(index_name)
    
    print(f"💾 Saving index to: {index_dir}")
    
    try:
        index.storage_context.persist(persist_dir=str(index_dir))
        print(f"✅ Index saved successfully")
        return index_dir
        
    except Exception as e:
        print(f"❌ Error saving index: {e}")
        raise


def load_index(
    index_name: str,
    embed_model: Any,
    llm: Any = None,
) -> Any:
    """
    Load index from disk
    
    Args:
        index_name: Name of the index directory
        embed_model: Embedding model to use
        llm: LLM model to use (optional)
        
    Returns:
        Loaded index object
    """
    index_dir = Config.get_index_dir(index_name)
    
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")
    
    print(f"📂 Loading index from: {index_dir}")
    
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=str(index_dir)
        )
        
        from llama_index.core import Settings
        Settings.embed_model = embed_model
        if llm:
            Settings.llm = llm
        
        index = load_index_from_storage(storage_context)
        
        print(f"✅ Index loaded successfully")
        return index
        
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        raise


def check_index_exists(index_name: str) -> bool:
    """
    Check if an index exists on disk
    
    Args:
        index_name: Name of the index directory
        
    Returns:
        True if index exists
    """
    index_dir = Config.get_index_dir(index_name)
    return index_dir.exists() and any(index_dir.iterdir())


def format_retrieved_nodes(nodes: List[NodeWithScore]) -> str:
    """
    Format retrieved nodes for display
    
    Args:
        nodes: List of nodes with scores
        
    Returns:
        Formatted string
    """
    formatted = []
    
    for i, node in enumerate(nodes, 1):
        score = node.score if node.score else 0.0
        text = node.node.get_content()[:200]  # First 200 chars
        
        formatted.append(
            f"📄 Node {i} (Score: {score:.4f})\n"
            f"{text}...\n"
        )
    
    return "\n".join(formatted)


def save_eval_results(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save evaluation results to JSON
    
    Args:
        results: Evaluation results dictionary
        output_path: Output file path (default: storage/eval_results.json)
        
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = Config.STORAGE_DIR / "eval_results.json"
    
    print(f"💾 Saving evaluation results to: {output_path}")
    
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved successfully")
        return output_path
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        raise


def load_eval_results(
    input_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load evaluation results from JSON
    
    Args:
        input_path: Input file path (default: storage/eval_results.json)
        
    Returns:
        Evaluation results dictionary
    """
    if input_path is None:
        input_path = Config.STORAGE_DIR / "eval_results.json"
    
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    print(f"📂 Loading evaluation results from: {input_path}")
    
    try:
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        print(f"✅ Results loaded successfully")
        return results
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        raise


def create_sample_eval_questions(output_path: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    Create sample evaluation questions
    
    Args:
        output_path: Where to save questions (default: data/eval_questions.json)
        
    Returns:
        List of question dictionaries
    """
    questions = [
        {
            "id": "q1",
            "question": "What is the main topic of the document?",
            "context": "general"
        },
        {
            "id": "q2",
            "question": "What are the key findings mentioned?",
            "context": "specific"
        },
        {
            "id": "q3",
            "question": "Can you summarize the methodology used?",
            "context": "detailed"
        },
        {
            "id": "q4",
            "question": "What are the conclusions?",
            "context": "summary"
        },
        {
            "id": "q5",
            "question": "What recommendations are provided?",
            "context": "actionable"
        },
    ]
    
    if output_path is None:
        output_path = Config.DATA_DIR / "eval_questions.json"
    
    print(f"💾 Saving sample questions to: {output_path}")
    
    try:
        with open(output_path, 'w') as f:
            json.dump(questions, f, indent=2)
        
        print(f"✅ Sample questions created")
        return questions
        
    except Exception as e:
        print(f"❌ Error creating sample questions: {e}")
        raise


def print_query_results(
    query: str,
    response: Any,
    retrieved_nodes: Optional[List[NodeWithScore]] = None,
):
    """
    Pretty print query results
    
    Args:
        query: The query string
        response: Response object
        retrieved_nodes: Retrieved nodes (optional)
    """
    print("\n" + "="*80)
    print(f"🔍 Query: {query}")
    print("="*80)
    
    print(f"\n💬 Response:\n{response}")
    
    if retrieved_nodes:
        print("\n" + "-"*80)
        print("📚 Retrieved Context:")
        print("-"*80)
        print(format_retrieved_nodes(retrieved_nodes))
    
    print("="*80 + "\n")


def get_file_stats(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get statistics about files in data directory
    
    Args:
        data_dir: Directory to analyze (default: Config.DATA_DIR)
        
    Returns:
        Dictionary with file statistics
    """
    data_dir = data_dir or Config.DATA_DIR
    
    if not data_dir.exists():
        return {"error": "Directory not found"}
    
    stats = {
        "total_files": 0,
        "file_types": {},
        "total_size_mb": 0.0,
    }
    
    for file_path in data_dir.rglob("*"):
        if file_path.is_file():
            stats["total_files"] += 1
            
            # File type
            ext = file_path.suffix or "no_extension"
            stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
            
            # Size
            stats["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
    
    return stats


# Convenience function to setup everything
def setup_rag_system():
    """
    Setup complete RAG system with all components
    
    Returns:
        Dictionary with llm, embed_model, reranker
    """
    from src.llm import get_llm
    from src.embeddings import get_embed_model, get_reranker
    
    print("\n🚀 Setting up RAG system...")
    
    # Validate config
    Config.validate_config()
    
    # Initialize components
    llm = get_llm()
    embed_model = get_embed_model()
    reranker = get_reranker()
    
    print("\n✅ RAG system ready!")
    
    return {
        "llm": llm,
        "embed_model": embed_model,
        "reranker": reranker,
    }


if __name__ == "__main__":
    # Test utilities
    print("🧪 Testing utilities...")
    
    # Print config
    Config.print_config()
    
    # Get file stats
    print("\n📊 File Statistics:")
    stats = get_file_stats()
    print(f"  Total files: {stats.get('total_files', 0)}")
    print(f"  File types: {stats.get('file_types', {})}")
    print(f"  Total size: {stats.get('total_size_mb', 0):.2f} MB")
    
    # Create sample questions
    print("\n📝 Creating sample evaluation questions...")
    create_sample_eval_questions()