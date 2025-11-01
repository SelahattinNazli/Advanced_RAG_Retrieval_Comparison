#!/usr/bin/env python3
"""
Advanced RAG Comparison Project Setup Script
Creates the complete project structure
"""

import os
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure"""
    
    # Base structure
    structure = {
        "src": {
            "__init__.py": "",
            "config.py": "",
            "embeddings.py": "",
            "llm.py": "",
            "utils.py": "",
            "retrievers": {
                "__init__.py": "",
                "basic_retriever.py": "",
                "sentence_window_retriever.py": "",
                "auto_merging_retriever.py": "",
            },
            "evaluation": {
                "__init__.py": "",
                "metrics.py": "",
                "evaluator.py": "",
            }
        },
        "data": {
            ".gitkeep": "",
            "README.md": "# Data Directory\n\nPlace your PDF/TXT files here for processing.\n"
        },
        "notebooks": {
            "experiment.ipynb": "",
        },
        "tests": {
            "__init__.py": "",
        }
    }
    
    # Root files
    root_files = {
        "streamlit_app.py": "",
        "pyproject.toml": "",
        ".env.example": "",
        ".gitignore": "",
        "README.md": "",
    }
    
    def create_structure(base_path, structure_dict):
        """Recursively create directory structure"""
        for name, content in structure_dict.items():
            path = base_path / name
            
            if isinstance(content, dict):
                # It's a directory
                path.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {path}")
                create_structure(path, content)
            else:
                # It's a file
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_text(content)
                    print(f"Created file: {path}")
    
    # Create project root
    project_root = Path("advanced-rag-comparison")
    project_root.mkdir(exist_ok=True)
    print(f"\n📁 Creating project: {project_root}\n")
    
    # Create structure
    create_structure(project_root, structure)
    
    # Create root files
    for filename, content in root_files.items():
        filepath = project_root / filename
        if not filepath.exists():
            filepath.write_text(content)
            print(f"Created file: {filepath}")
    
    print(f"\n✅ Project structure created successfully!\n")
    print("Next steps:")
    print("  cd advanced-rag-comparison")
    print("  uv venv")
    print("  source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows")
    print("  uv pip install -r requirements.txt")

if __name__ == "__main__":
    create_project_structure()