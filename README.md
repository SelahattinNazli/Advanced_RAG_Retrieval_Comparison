# Advanced RAG Comparison

A comprehensive framework for comparing and evaluating different Retrieval-Augmented Generation (RAG) methods. This project helps researchers and developers understand which retrieval strategy works best for their specific use case by providing side-by-side comparisons with quantitative metrics.

## Why This Project?

When building RAG applications, choosing the right retrieval strategy significantly impacts answer quality. However, comparing different approaches requires:
- Setting up multiple retrieval pipelines
- Implementing evaluation metrics
- Running comparative experiments
- Visualizing results

This project solves these challenges by providing a ready-to-use comparison framework that:
- **Saves Development Time**: Pre-implemented retrieval methods and evaluation pipeline
- **Enables Data-Driven Decisions**: Quantitative metrics (faithfulness, relevance, context quality) help you choose objectively
- **Supports Experimentation**: Easy to test with your own documents and questions
- **Completely Free**: Uses open-source models (Ollama + HuggingFace) with no API costs

## What's Inside?

### Three Retrieval Methods

1. **Basic RAG** (Baseline)
   - Simple chunking strategy
   - Fast and straightforward
   - Good starting point for most applications

2. **Sentence Window Retrieval**
   - Retrieves small chunks but includes surrounding context
   - Better context preservation
   - Configurable window size (1, 3, 5, 7 sentences)

3. **Auto-Merging Retrieval**
   - Hierarchical chunking with automatic context merging
   - Balances granularity and context
   - Adaptive to query complexity

### Evaluation Metrics (RAGAS)

Each retrieval method is evaluated on:
- **Faithfulness**: How factually accurate are the answers?
- **Answer Relevancy**: How relevant is the response to the question?
- **Context Relevancy**: How relevant are the retrieved documents?

### Interactive Dashboard

A Streamlit interface that allows you to:
- Upload your own documents (PDF/TXT)
- Ask questions and see responses from all three methods
- Compare metrics side-by-side
- View retrieved context for each method
- Get recommendations on which method performs best

## Use Cases

This project is useful if you:
- Are building a RAG application and need to choose a retrieval strategy
- Want to understand how different chunking approaches affect answer quality
- Need to benchmark retrieval methods on your specific domain
- Are researching RAG optimization techniques
- Want to learn about advanced retrieval patterns

## Technology Stack

- **LLM**: Ollama (Qwen 2.5:1.7b) - Local, private, no API costs
- **Embeddings**: HuggingFace (bge-small-en-v1.5) - Free, open-source
- **Evaluation**: RAGAS - Industry-standard RAG metrics
- **Framework**: LlamaIndex - Advanced retrieval patterns
- **Interface**: Streamlit - Interactive comparison dashboard

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **UV** (Python package manager)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Ollama** with Qwen model
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull Qwen model
   ollama pull qwen2.5:1.7b
   ```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/advanced-rag-comparison.git
cd advanced-rag-comparison

# Create virtual environment with UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"
```

### Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env if needed (defaults should work)
```

### Add Your Documents

Place PDF or TXT files in the `data/` directory:

```bash
cp your_document.pdf data/
```

### Run the Application

```bash
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` in your browser.

## Project Structure

```
advanced-rag-comparison/
├── src/
│   ├── config.py              # Configuration management
│   ├── embeddings.py          # HuggingFace embeddings
│   ├── llm.py                 # Ollama integration
│   ├── retrievers/            # Retrieval implementations
│   │   ├── basic_retriever.py
│   │   ├── sentence_window_retriever.py
│   │   └── auto_merging_retriever.py
│   ├── evaluation/            # RAGAS evaluation
│   │   ├── metrics.py
│   │   └── evaluator.py
│   └── utils.py               # Helper functions
├── data/                      # Your documents
├── notebooks/                 # Jupyter experiments
│   └── experiment.ipynb
├── tests/                     # Test suite
├── streamlit_app.py           # Main dashboard
└── pyproject.toml            # UV configuration
```

## Using the Dashboard

### Tab 1: Query & Compare
1. Choose document source (existing files or upload new)
2. Enter your question
3. Click "Run Comparison"
4. View answers from all three methods side-by-side
5. Compare metrics and response times
6. See which method works best for your query

### Tab 2: Benchmark Results
- View pre-computed evaluation results
- See aggregate metrics across multiple questions
- Compare performance with charts and tables
- Identify the best-performing method for your use case

## Running Experiments

Use the provided Jupyter notebook for detailed experimentation:

```bash
jupyter notebook notebooks/experiment.ipynb
```

The notebook allows you to:
- Test different window sizes (1, 3, 5, 7, 9)
- Experiment with various chunk size hierarchies
- Run comprehensive evaluations
- Generate benchmark results for the dashboard

## Configuration Options

Edit `.env` to customize behavior:

```bash
# Model Configuration
OLLAMA_MODEL=qwen2.5:1.7b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Retrieval Parameters
CHUNK_SIZE=512
CHUNK_OVERLAP=50
SIMILARITY_TOP_K=6

# Sentence Window
SENTENCE_WINDOW_SIZE=3

# Auto-Merging
AUTO_MERGE_CHUNK_SIZES=2048,512,128
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_config.py -v
```

## Example Results

Typical performance on technical documentation:

| Method | Faithfulness | Answer Relevancy | Avg Response Time |
|--------|--------------|------------------|-------------------|
| Basic RAG | 0.75 | 0.70 | 2.3s |
| Sentence Window | 0.82 | 0.78 | 3.1s |
| Auto-Merging | 0.85 | 0.80 | 2.8s |

Results vary based on document type and query complexity.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional retrieval methods (hybrid search, graph-based, etc.)
- More evaluation metrics
- Support for additional file formats
- Performance optimizations
- Documentation improvements

Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - feel free to use for any purpose.

## Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [RAGAS](https://github.com/explodinggradients/ragas) - Evaluation metrics
- [Ollama](https://ollama.com/) - Local LLM runtime

