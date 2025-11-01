"""
Evaluator orchestrator for comparing multiple retrievers
"""
import time
from typing import List, Dict, Any, Optional
import pandas as pd
from src.evaluation.metrics import RAGASEvaluator, compute_all_metrics
from src.config import Config


class RetrieverEvaluator:
    """Orchestrates evaluation of multiple retriever systems"""
    
    def __init__(self, ragas_evaluator: Optional[RAGASEvaluator] = None):
        """
        Initialize evaluator
        
        Args:
            ragas_evaluator: RAGAS evaluator instance (will create if None)
        """
        self.ragas_evaluator = ragas_evaluator or RAGASEvaluator()
        self.results = []
    
    def evaluate_retriever(
        self,
        retriever: Any,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a single retriever on a set of questions
        
        Args:
            retriever: Retriever instance with query() method
            questions: List of questions to evaluate
            ground_truths: Optional ground truth answers
            verbose: Print progress
            
        Returns:
            Dictionary with evaluation results
        """
        retriever_name = retriever.get_retriever_name()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🔍 Evaluating: {retriever_name}")
            print(f"{'='*70}")
            print(f"Questions: {len(questions)}")
        
        answers = []
        contexts_list = []
        response_times = []
        
        # Query each question
        for i, question in enumerate(questions, 1):
            if verbose:
                print(f"\n[{i}/{len(questions)}] Processing: {question[:50]}...")
            
            try:
                start_time = time.time()
                response, nodes = retriever.query(question, return_nodes=True)
                elapsed_time = time.time() - start_time
                
                # Extract answer and contexts
                answer = str(response)
                contexts = [node.node.get_content() for node in nodes]
                
                answers.append(answer)
                contexts_list.append(contexts)
                response_times.append(elapsed_time)
                
                if verbose:
                    print(f"   ⏱️  Time: {elapsed_time:.2f}s")
                    print(f"   📚 Contexts: {len(contexts)}")
                    print(f"   💬 Answer: {answer[:100]}...")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                answers.append("")
                contexts_list.append([])
                response_times.append(0.0)
        
        # Compute metrics
        if verbose:
            print(f"\n📊 Computing metrics...")
        
        try:
            ragas_results = self.ragas_evaluator.evaluate_dataset(
                questions=questions,
                answers=answers,
                contexts=contexts_list,
                ground_truths=ground_truths,
            )
            
            metrics_summary = ragas_results["summary"]
            
        except Exception as e:
            print(f"⚠️  RAGAS evaluation failed: {e}")
            metrics_summary = {}
        
        # Add custom metrics
        metrics_summary["avg_response_time"] = sum(response_times) / len(response_times)
        metrics_summary["total_time"] = sum(response_times)
        
        # Aggregate results
        result = {
            "retriever_name": retriever_name,
            "retriever_config": retriever.get_config(),
            "num_questions": len(questions),
            "metrics": metrics_summary,
            "questions": questions,
            "answers": answers,
            "contexts": contexts_list,
            "response_times": response_times,
        }
        
        if ground_truths:
            result["ground_truths"] = ground_truths
        
        self.results.append(result)
        
        if verbose:
            self._print_result_summary(result)
        
        return result
    
    def evaluate_multiple_retrievers(
        self,
        retrievers: List[Any],
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple retrievers on the same questions
        
        Args:
            retrievers: List of retriever instances
            questions: List of questions
            ground_truths: Optional ground truth answers
            verbose: Print progress
            
        Returns:
            List of evaluation results
        """
        print(f"\n{'='*70}")
        print(f"🚀 Evaluating {len(retrievers)} Retrievers")
        print(f"{'='*70}")
        
        all_results = []
        
        for retriever in retrievers:
            result = self.evaluate_retriever(
                retriever=retriever,
                questions=questions,
                ground_truths=ground_truths,
                verbose=verbose,
            )
            all_results.append(result)
        
        if verbose:
            self._print_comparison(all_results)
        
        return all_results
    
    def _print_result_summary(self, result: Dict[str, Any]):
        """Print summary of single result"""
        print(f"\n{'='*70}")
        print(f"📊 Results Summary: {result['retriever_name']}")
        print(f"{'='*70}")
        
        metrics = result["metrics"]
        
        # RAGAS metrics
        ragas_metrics = ["faithfulness", "answer_relevancy", "context_relevancy"]
        print("\n📈 RAGAS Metrics:")
        for metric in ragas_metrics:
            if metric in metrics:
                score = metrics[metric]
                print(f"   {metric:20s}: {score:.4f} {'🟢' if score > 0.7 else '🟡' if score > 0.5 else '🔴'}")
        
        # Performance metrics
        print("\n⏱️  Performance:")
        print(f"   Avg Response Time    : {metrics.get('avg_response_time', 0):.2f}s")
        print(f"   Total Time           : {metrics.get('total_time', 0):.2f}s")
        
        print(f"\n{'='*70}")
    
    def _print_comparison(self, results: List[Dict[str, Any]]):
        """Print comparison of all results"""
        print(f"\n{'='*70}")
        print(f"📊 Retriever Comparison")
        print(f"{'='*70}\n")
        
        # Create comparison table
        comparison_data = []
        
        for result in results:
            row = {
                "Retriever": result["retriever_name"],
            }
            
            # Add metrics
            metrics = result["metrics"]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = value
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by faithfulness (or first available metric)
        if "faithfulness" in df.columns:
            df = df.sort_values("faithfulness", ascending=False)
        
        print(df.to_string(index=False))
        print(f"\n{'='*70}")
        
        # Highlight best performer
        if "faithfulness" in df.columns:
            best = df.iloc[0]
            print(f"\n🏆 Best Performer: {best['Retriever']}")
            print(f"   Faithfulness: {best['faithfulness']:.4f}")
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Get comparison results as DataFrame
        
        Returns:
            DataFrame with all results
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for result in self.results:
            row = {
                "retriever": result["retriever_name"],
                **result["metrics"]
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def save_results(self, output_path: str):
        """Save results to file"""
        from src.utils import save_eval_results
        
        # Prepare results for saving (remove non-serializable objects)
        serializable_results = []
        for result in self.results:
            clean_result = {
                "retriever_name": result["retriever_name"],
                "retriever_config": result["retriever_config"],
                "num_questions": result["num_questions"],
                "metrics": result["metrics"],
                "questions": result["questions"],
                "answers": result["answers"],
                "response_times": result["response_times"],
            }
            serializable_results.append(clean_result)
        
        save_eval_results(
            {"results": serializable_results},
            output_path=output_path
        )


def compare_retrievers(
    retrievers: List[Any],
    questions: List[str],
    ground_truths: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convenience function to compare multiple retrievers
    
    Args:
        retrievers: List of retriever instances
        questions: Questions to evaluate
        ground_truths: Optional ground truth answers
        
    Returns:
        DataFrame with comparison results
    """
    evaluator = RetrieverEvaluator()
    evaluator.evaluate_multiple_retrievers(
        retrievers=retrievers,
        questions=questions,
        ground_truths=ground_truths,
    )
    
    return evaluator.get_comparison_dataframe()


if __name__ == "__main__":
    # Test evaluator
    from src.utils import load_documents, setup_rag_system
    from src.retrievers import (
        build_basic_retriever,
        build_sentence_window_retriever,
    )
    
    print("🧪 Testing Retriever Evaluator...")
    
    # Setup
    system = setup_rag_system()
    documents = load_documents()
    
    if documents:
        # Build retrievers
        basic = build_basic_retriever(
            documents=documents,
            llm=system["llm"],
            embed_model=system["embed_model"],
            reranker=system["reranker"],
            force_rebuild=True,
        )
        
        window = build_sentence_window_retriever(
            documents=documents,
            llm=system["llm"],
            embed_model=system["embed_model"],
            reranker=system["reranker"],
            window_size=3,
            force_rebuild=True,
        )
        
        # Test questions
        questions = [
            "What is the main topic?",
            "What are the key points?",
        ]
        
        # Compare
        df = compare_retrievers(
            retrievers=[basic, window],
            questions=questions,
        )
        
        print("\n📊 Comparison Results:")
        print(df)