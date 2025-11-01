"""
Evaluation metrics using RAGAS
"""
from typing import List, Dict, Any, Optional
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Try to import context_relevancy (old name) or context_relevance (new name)
try:
    from ragas.metrics import context_relevancy
except ImportError:
    try:
        from ragas.metrics import context_relevance as context_relevancy
    except ImportError:
        context_relevancy = None
from langchain_community.llms import Ollama as LangChainOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import Config


class RAGASEvaluator:
    """RAGAS-based evaluation for RAG systems"""
    
    def __init__(
        self,
        llm_model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ):
        """
        Initialize RAGAS evaluator
        
        Args:
            llm_model: Ollama model name
            embed_model: HuggingFace embedding model name
        """
        self.llm_model = llm_model or Config.OLLAMA_MODEL
        self.embed_model_name = embed_model or Config.EMBEDDING_MODEL
        
        print(f"🔧 Initializing RAGAS Evaluator...")
        print(f"   LLM: {self.llm_model}")
        print(f"   Embeddings: {self.embed_model_name}")
        
        # Initialize LangChain Ollama for RAGAS
        self.llm = LangChainOllama(
            model=self.llm_model,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=Config.OLLAMA_TEMPERATURE,
        )
        
        # Initialize embeddings for RAGAS
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
            model_kwargs={'device': Config.EMBEDDING_DEVICE},
        )
        
        print(f"✅ RAGAS Evaluator initialized")
    
    def evaluate_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system using RAGAS metrics
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved contexts (list of strings for each question)
            ground_truths: Optional list of ground truth answers
            metrics: List of RAGAS metrics to compute (default: faithfulness, answer_relevancy, context_relevancy)
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n📊 Evaluating with RAGAS...")
        print(f"   Questions: {len(questions)}")
        print(f"   Answers: {len(answers)}")
        print(f"   Contexts: {len(contexts)}")
        
        # Default metrics
        if metrics is None:
            metrics = [
                faithfulness,
                answer_relevancy,
            ]
            
            # Add context_relevancy if available
            if context_relevancy:
                metrics.append(context_relevancy)
            
            # Add context precision and recall if ground truths provided
            if ground_truths:
                metrics.extend([context_precision, context_recall])
        
        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        # Run evaluation
        try:
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )
            
            print(f"✅ Evaluation completed")
            
            # Convert to dict
            results_dict = {
                "scores": result.to_pandas().to_dict('records'),
                "summary": {
                    metric.name: float(result[metric.name])
                    for metric in metrics
                }
            }
            
            return results_dict
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise
    
    def evaluate_single_query(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single query-answer pair
        
        Args:
            question: Query question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary of metric scores
        """
        result = self.evaluate_dataset(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None,
        )
        
        return result["summary"]
    
    def print_results(self, results: Dict[str, Any]):
        """Pretty print evaluation results"""
        print("\n" + "="*60)
        print("📊 RAGAS Evaluation Results")
        print("="*60)
        
        if "summary" in results:
            print("\n📈 Metric Scores:")
            for metric, score in results["summary"].items():
                print(f"   {metric:20s}: {score:.4f}")
        
        print("\n" + "="*60)


class CustomMetrics:
    """Custom metrics for additional evaluation"""
    
    @staticmethod
    def response_length(answer: str) -> int:
        """Get response length in words"""
        return len(answer.split())
    
    @staticmethod
    def context_utilization(answer: str, contexts: List[str]) -> float:
        """
        Calculate what percentage of contexts appear in answer
        
        Args:
            answer: Generated answer
            contexts: Retrieved contexts
            
        Returns:
            Utilization ratio (0-1)
        """
        if not contexts:
            return 0.0
        
        answer_lower = answer.lower()
        utilized = 0
        
        for context in contexts:
            # Check if significant words from context appear in answer
            context_words = set(context.lower().split())
            # Filter out common words
            significant_words = {
                w for w in context_words 
                if len(w) > 4 and w.isalnum()
            }
            
            if significant_words:
                overlap = sum(1 for w in significant_words if w in answer_lower)
                if overlap / len(significant_words) > 0.3:  # 30% overlap threshold
                    utilized += 1
        
        return utilized / len(contexts)
    
    @staticmethod
    def retrieval_efficiency(
        num_contexts: int,
        context_utilization: float,
    ) -> float:
        """
        Calculate retrieval efficiency score
        
        Args:
            num_contexts: Number of contexts retrieved
            context_utilization: Utilization ratio
            
        Returns:
            Efficiency score (0-1)
        """
        if num_contexts == 0:
            return 0.0
        
        # Balance between having enough contexts and using them efficiently
        ideal_contexts = 3
        context_penalty = abs(num_contexts - ideal_contexts) / ideal_contexts
        
        return context_utilization * (1 - min(context_penalty, 0.5))


def compute_all_metrics(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
    ragas_evaluator: Optional[RAGASEvaluator] = None,
) -> Dict[str, Any]:
    """
    Compute all available metrics for a query
    
    Args:
        question: Query question
        answer: Generated answer
        contexts: Retrieved contexts
        ground_truth: Optional ground truth answer
        ragas_evaluator: RAGAS evaluator instance
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # RAGAS metrics
    if ragas_evaluator:
        try:
            ragas_scores = ragas_evaluator.evaluate_single_query(
                question, answer, contexts, ground_truth
            )
            metrics.update(ragas_scores)
        except Exception as e:
            print(f"⚠️  RAGAS evaluation failed: {e}")
    
    # Custom metrics
    metrics["response_length"] = CustomMetrics.response_length(answer)
    metrics["num_contexts"] = len(contexts)
    
    context_util = CustomMetrics.context_utilization(answer, contexts)
    metrics["context_utilization"] = context_util
    
    metrics["retrieval_efficiency"] = CustomMetrics.retrieval_efficiency(
        len(contexts), context_util
    )
    
    return metrics


if __name__ == "__main__":
    # Test RAGAS evaluator
    print("🧪 Testing RAGAS Evaluator...")
    
    evaluator = RAGASEvaluator()
    
    # Sample data
    questions = ["What is RAG?", "How does retrieval work?"]
    answers = [
        "RAG stands for Retrieval Augmented Generation.",
        "Retrieval works by finding relevant documents."
    ]
    contexts = [
        ["RAG is a technique that combines retrieval with generation."],
        ["Documents are retrieved using similarity search."]
    ]
    
    # Evaluate
    results = evaluator.evaluate_dataset(questions, answers, contexts)
    evaluator.print_results(results)
    
    # Test custom metrics
    print("\n🧪 Testing Custom Metrics...")
    custom_results = compute_all_metrics(
        question=questions[0],
        answer=answers[0],
        contexts=contexts[0],
    )
    print(f"Custom metrics: {custom_results}")