"""
Streamlit Dashboard for Advanced RAG Comparison
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
import json
import tempfile

from src.config import Config
from src.utils import load_documents, setup_rag_system, load_eval_results
from src.retrievers import (
    build_basic_retriever,
    build_sentence_window_retriever,
    build_auto_merging_retriever,
)
from src.evaluation import RAGASEvaluator, compute_all_metrics


# Page config
st.set_page_config(
    page_title="Advanced RAG Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 24px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize RAG system components"""
    with st.spinner("🔧 Initializing RAG system..."):
        system = setup_rag_system()
        ragas_evaluator = RAGASEvaluator()
    return system, ragas_evaluator


def load_uploaded_document(uploaded_file):
    """Load document from uploaded file"""
    from llama_index.core import SimpleDirectoryReader
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file to temp directory
        temp_path = Path(temp_dir) / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getvalue())
        
        # Load document
        reader = SimpleDirectoryReader(input_dir=temp_dir)
        documents = reader.load_data()
        
    return documents


def build_retrievers_from_docs(documents, system):
    """Build all three retrievers from documents"""
    with st.spinner("🔨 Building retrievers..."):
        # Basic RAG
        basic = build_basic_retriever(
            documents=documents,
            llm=system["llm"],
            embed_model=system["embed_model"],
            reranker=system["reranker"],
            index_name="temp_basic",
            force_rebuild=True,
        )
        
        # Sentence Window
        window = build_sentence_window_retriever(
            documents=documents,
            llm=system["llm"],
            embed_model=system["embed_model"],
            reranker=system["reranker"],
            window_size=Config.SENTENCE_WINDOW_SIZE,
            index_name="temp_window",
            force_rebuild=True,
        )
        
        # Auto-Merging
        auto_merge = build_auto_merging_retriever(
            documents=documents,
            llm=system["llm"],
            embed_model=system["embed_model"],
            reranker=system["reranker"],
            chunk_sizes=Config.AUTO_MERGE_CHUNK_SIZES,
            index_name="temp_auto",
            force_rebuild=True,
        )
    
    return {
        "Basic RAG": basic,
        "Sentence Window": window,
        "Auto-Merging": auto_merge,
    }


def query_and_compare_tab(retrievers, ragas_evaluator):
    """Tab 1: Real-time query comparison"""
    st.header("🔍 Query & Compare")
    st.markdown("Test your question against all three retrieval methods in real-time.")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="What is the main topic of the document?",
        key="query_input"
    )
    
    if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
        if not query:
            st.warning("⚠️ Please enter a question first!")
            return
        
        st.markdown("---")
        
        # Create three columns for comparison
        cols = st.columns(3)
        
        results = {}
        
        # Query each retriever
        for idx, (name, retriever) in enumerate(retrievers.items()):
            with cols[idx]:
                st.subheader(f"{'🔷' if idx == 0 else '🟢' if idx == 1 else '🟣'} {name}")
                
                with st.spinner(f"Querying {name}..."):
                    start_time = time.time()
                    
                    try:
                        response, nodes = retriever.query(query, return_nodes=True)
                        elapsed_time = time.time() - start_time
                        
                        answer = str(response)
                        contexts = [node.node.get_content() for node in nodes]
                        
                        # Display answer
                        st.markdown("**💬 Answer:**")
                        st.info(answer)
                        
                        # Compute metrics
                        with st.spinner("Computing metrics..."):
                            try:
                                metrics = compute_all_metrics(
                                    question=query,
                                    answer=answer,
                                    contexts=contexts,
                                    ragas_evaluator=ragas_evaluator,
                                )
                            except Exception as e:
                                st.warning(f"⚠️ Metrics computation failed: {e}")
                                metrics = {"response_time": elapsed_time}
                        
                        # Display metrics
                        st.markdown("**📊 Metrics:**")
                        
                        # RAGAS metrics
                        if "faithfulness" in metrics:
                            st.metric("Faithfulness", f"{metrics['faithfulness']:.3f}")
                        if "answer_relevancy" in metrics:
                            st.metric("Answer Relevancy", f"{metrics['answer_relevancy']:.3f}")
                        
                        # Performance
                        st.metric("⏱️ Response Time", f"{elapsed_time:.2f}s")
                        st.metric("📚 Contexts Retrieved", len(contexts))
                        
                        # Store results
                        results[name] = {
                            "answer": answer,
                            "contexts": contexts,
                            "metrics": metrics,
                            "time": elapsed_time,
                        }
                        
                        # Show contexts in expander
                        with st.expander("📖 View Retrieved Contexts"):
                            for i, context in enumerate(contexts[:3], 1):  # Show top 3
                                st.markdown(f"**Context {i}:**")
                                st.text(context[:300] + "..." if len(context) > 300 else context)
                                st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        # Summary comparison
        if len(results) > 1:
            st.markdown("---")
            st.subheader("📊 Quick Comparison")
            
            comparison_data = []
            for name, result in results.items():
                row = {"Retriever": name}
                if "faithfulness" in result["metrics"]:
                    row["Faithfulness"] = f"{result['metrics']['faithfulness']:.3f}"
                if "answer_relevancy" in result["metrics"]:
                    row["Answer Relevancy"] = f"{result['metrics']['answer_relevancy']:.3f}"
                row["Response Time"] = f"{result['time']:.2f}s"
                row["Contexts"] = len(result["contexts"])
                comparison_data.append(row)
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Winner recommendation
            st.markdown("---")
            st.subheader("🏆 Recommendation")
            if "faithfulness" in results[list(results.keys())[0]]["metrics"]:
                best_method = max(
                    results.items(),
                    key=lambda x: x[1]["metrics"].get("faithfulness", 0)
                )
                st.success(f"**Best method for this query:** {best_method[0]} " +
                          f"(Faithfulness: {best_method[1]['metrics']['faithfulness']:.3f})")


def benchmark_results_tab():
    """Tab 2: Pre-computed benchmark results"""
    st.header("📊 Benchmark Results")
    st.markdown("Performance comparison on test dataset")
    
    # Check if results exist
    results_path = Config.STORAGE_DIR / "eval_results.json"
    
    if not results_path.exists():
        st.info("📝 No benchmark results found. Run the evaluation notebook first!")
        
        st.markdown("### How to generate benchmark results:")
        st.code("""
# In notebooks/experiment.ipynb or Python script:

from src.evaluation import RetrieverEvaluator

evaluator = RetrieverEvaluator()
results = evaluator.evaluate_multiple_retrievers(
    retrievers=[basic, window, auto_merge],
    questions=test_questions,
)

evaluator.save_results("storage/eval_results.json")
        """, language="python")
        
        # Create sample results for demo
        if st.button("📊 Generate Sample Results (Demo)"):
            sample_results = {
                "results": [
                    {
                        "retriever_name": "Basic RAG",
                        "metrics": {
                            "faithfulness": 0.75,
                            "answer_relevancy": 0.70,
                            "avg_response_time": 2.3,
                        }
                    },
                    {
                        "retriever_name": "Sentence Window (size=3)",
                        "metrics": {
                            "faithfulness": 0.82,
                            "answer_relevancy": 0.78,
                            "avg_response_time": 3.1,
                        }
                    },
                    {
                        "retriever_name": "Auto-Merging",
                        "metrics": {
                            "faithfulness": 0.85,
                            "answer_relevancy": 0.80,
                            "avg_response_time": 2.8,
                        }
                    },
                ]
            }
            
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(sample_results, f, indent=2)
            
            st.success("✅ Sample results generated!")
            st.rerun()
        
        return
    
    # Load results
    try:
        results = load_eval_results(results_path)
        results_list = results.get("results", [])
        
        if not results_list:
            st.warning("⚠️ Results file is empty")
            return
        
        # Metrics comparison
        st.subheader("📈 Overall Performance Metrics")
        
        metrics_to_plot = ["faithfulness", "answer_relevancy"]
        metric_data = []
        
        for result in results_list:
            for metric in metrics_to_plot:
                if metric in result["metrics"]:
                    metric_data.append({
                        "Retriever": result["retriever_name"],
                        "Metric": metric.replace("_", " ").title(),
                        "Score": result["metrics"][metric]
                    })
        
        if metric_data:
            df_metrics = pd.DataFrame(metric_data)
            
            # Bar chart
            fig = px.bar(
                df_metrics,
                x="Metric",
                y="Score",
                color="Retriever",
                barmode="group",
                title="Metric Comparison Across Retrievers",
                color_discrete_sequence=["#636EFA", "#00CC96", "#AB63FA"]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("📋 Detailed Results Table")
        
        table_data = []
        for result in results_list:
            row = {"Retriever": result["retriever_name"]}
            row.update(result["metrics"])
            table_data.append(row)
        
        df_table = pd.DataFrame(table_data)
        
        # Format numeric columns
        for col in df_table.columns:
            if col != "Retriever" and df_table[col].dtype in ['float64', 'int64']:
                df_table[col] = df_table[col].round(4)
        
        st.dataframe(df_table, use_container_width=True)
        
        # Winner
        st.subheader("🏆 Best Performer")
        
        if "faithfulness" in df_table.columns:
            best_idx = df_table["faithfulness"].idxmax()
            best_retriever = df_table.loc[best_idx, "Retriever"]
            best_score = df_table.loc[best_idx, "faithfulness"]
            
            st.success(f"**{best_retriever}** with Faithfulness score of **{best_score:.4f}**")
    
    except Exception as e:
        st.error(f"❌ Error loading results: {e}")


def main():
    """Main Streamlit app"""
    
    # Title
    st.title("🚀 Advanced RAG Comparison")
    st.markdown("Compare **Basic RAG**, **Sentence Window**, and **Auto-Merging** retrieval methods")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Document source selection
        st.markdown("### 📄 Document Source")
        doc_source = st.radio(
            "Choose:",
            ["📁 Use data/ folder", "📤 Upload document"],
            key="doc_source"
        )
        
        uploaded_file = None
        documents = None
        
        if doc_source == "📤 Upload document":
            st.markdown("**Upload PDF or TXT:**")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf', 'txt'],
                help="Upload a PDF or TXT file to test"
            )
            
            if uploaded_file:
                st.success(f"✅ {uploaded_file.name}")
                st.info("💡 This is temporary. To save permanently, add to `data/` folder.")
            else:
                st.warning("⚠️ Upload a document")
        else:
            # Show existing files
            data_files = list(Config.DATA_DIR.glob("*.pdf")) + list(Config.DATA_DIR.glob("*.txt"))
            if data_files:
                st.success(f"✅ {len(data_files)} file(s) found")
                with st.expander("📂 View files"):
                    for f in data_files:
                        st.text(f"• {f.name}")
            else:
                st.warning("⚠️ No files in data/ folder")
        
        st.markdown("---")
        
        # System info
        st.markdown("### 🤖 Models")
        st.code(f"LLM: {Config.OLLAMA_MODEL}")
        st.code(f"Embed: {Config.EMBEDDING_MODEL}")
        
        st.markdown("### 🔧 Settings")
        st.code(f"Window: {Config.SENTENCE_WINDOW_SIZE}")
        st.code(f"Chunks: {Config.AUTO_MERGE_CHUNK_SIZES}")
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        Compare 3 retrieval methods:
        - **Basic RAG**
        - **Sentence Window**
        - **Auto-Merging**
        
        Built with LlamaIndex, RAGAS & Ollama
        """)
    
    # Initialize system
    try:
        system, ragas_evaluator = initialize_system()
    except Exception as e:
        st.error(f"❌ System init failed: {e}")
        st.stop()
    
    # Load documents
    if doc_source == "📤 Upload document":
        if not uploaded_file:
            st.info("👈 Please upload a document from the sidebar")
            st.stop()
        try:
            with st.spinner("📄 Loading uploaded document..."):
                documents = load_uploaded_document(uploaded_file)
        except Exception as e:
            st.error(f"❌ Failed to load: {e}")
            st.stop()
    else:
        try:
            documents = load_documents()
            if not documents:
                st.warning("⚠️ No documents in data/ folder. Upload one or add files there.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Failed to load: {e}")
            st.stop()
    
    # Build retrievers
    try:
        retrievers = build_retrievers_from_docs(documents, system)
        st.success("✅ All retrievers ready!")
    except Exception as e:
        st.error(f"❌ Failed to build retrievers: {e}")
        st.stop()
    
    # Tabs
    tab1, tab2 = st.tabs(["🔍 Query & Compare", "📊 Benchmark Results"])
    
    with tab1:
        query_and_compare_tab(retrievers, ragas_evaluator)
    
    with tab2:
        benchmark_results_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Advanced RAG Comparison | Built with ❤️ using LlamaIndex, RAGAS & Ollama"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()