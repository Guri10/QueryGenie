"""
Evaluation Scripts for QueryGenie RAG System
Implements hit@k, latency, and other performance metrics
"""

import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import statistics
from pathlib import Path

from .faiss_manager import FAISSManager
from .rag_pipeline import RAGPipeline


@dataclass
class EvaluationResult:
    """Results from evaluation"""
    metric_name: str
    value: float
    details: Dict[str, Any]


@dataclass
class QueryResult:
    """Result from a single query"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float
    total_time: float
    relevance_scores: List[float] = None


class QueryGenieEvaluator:
    """Evaluator for QueryGenie RAG system"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize evaluator
        
        Args:
            rag_pipeline: Initialized RAG pipeline
        """
        self.rag_pipeline = rag_pipeline
        self.results = []
    
    def evaluate_latency(self, 
                        queries: List[str], 
                        k: int = 5,
                        num_runs: int = 3) -> EvaluationResult:
        """
        Evaluate latency metrics
        
        Args:
            queries: List of test queries
            k: Number of sources to retrieve
            num_runs: Number of runs per query for averaging
            
        Returns:
            EvaluationResult with latency metrics
        """
        print(f"Evaluating latency with {len(queries)} queries, {num_runs} runs each...")
        
        retrieval_times = []
        generation_times = []
        total_times = []
        
        for query in queries:
            query_retrieval_times = []
            query_generation_times = []
            query_total_times = []
            
            for run in range(num_runs):
                try:
                    start_time = time.time()
                    response = self.rag_pipeline.query(query, k=k)
                    
                    query_retrieval_times.append(response.retrieval_time)
                    query_generation_times.append(response.generation_time)
                    query_total_times.append(response.total_time)
                    
                except Exception as e:
                    print(f"Error evaluating query '{query}': {e}")
                    continue
            
            # Average across runs for this query
            if query_retrieval_times:
                retrieval_times.append(statistics.mean(query_retrieval_times))
                generation_times.append(statistics.mean(query_generation_times))
                total_times.append(statistics.mean(query_total_times))
        
        # Calculate overall statistics
        latency_stats = {
            "avg_retrieval_time": statistics.mean(retrieval_times) if retrieval_times else 0,
            "avg_generation_time": statistics.mean(generation_times) if generation_times else 0,
            "avg_total_time": statistics.mean(total_times) if total_times else 0,
            "median_retrieval_time": statistics.median(retrieval_times) if retrieval_times else 0,
            "median_generation_time": statistics.median(generation_times) if generation_times else 0,
            "median_total_time": statistics.median(total_times) if total_times else 0,
            "std_retrieval_time": statistics.stdev(retrieval_times) if len(retrieval_times) > 1 else 0,
            "std_generation_time": statistics.stdev(generation_times) if len(generation_times) > 1 else 0,
            "std_total_time": statistics.stdev(total_times) if len(total_times) > 1 else 0,
            "min_total_time": min(total_times) if total_times else 0,
            "max_total_time": max(total_times) if total_times else 0,
            "num_queries": len(queries),
            "num_successful": len(total_times)
        }
        
        return EvaluationResult(
            metric_name="latency",
            value=latency_stats["avg_total_time"],
            details=latency_stats
        )
    
    def evaluate_hit_at_k(self, 
                          queries: List[str], 
                          k_values: List[int] = [1, 3, 5, 10],
                          relevance_threshold: float = 0.7) -> List[EvaluationResult]:
        """
        Evaluate hit@k metrics
        
        Args:
            queries: List of test queries
            k_values: List of k values to evaluate
            relevance_threshold: Threshold for considering a result relevant
            
        Returns:
            List of EvaluationResult for each k value
        """
        print(f"Evaluating hit@k with {len(queries)} queries...")
        
        results = []
        
        for k in k_values:
            hits = 0
            total_queries = 0
            hit_details = []
            
            for query in queries:
                try:
                    # Get search results
                    sources = self.rag_pipeline.faiss_manager.search(query, k=k)
                    
                    if sources:
                        # Check if any result has similarity score above threshold
                        relevant_results = [s for s in sources if s.get('similarity_score', 0) >= relevance_threshold]
                        
                        if relevant_results:
                            hits += 1
                        
                        hit_details.append({
                            'query': query,
                            'hits': len(relevant_results),
                            'max_similarity': max([s.get('similarity_score', 0) for s in sources]),
                            'avg_similarity': statistics.mean([s.get('similarity_score', 0) for s in sources])
                        })
                    
                    total_queries += 1
                    
                except Exception as e:
                    print(f"Error evaluating hit@k for query '{query}': {e}")
                    continue
            
            hit_at_k = hits / total_queries if total_queries > 0 else 0
            
            results.append(EvaluationResult(
                metric_name=f"hit@{k}",
                value=hit_at_k,
                details={
                    "hits": hits,
                    "total_queries": total_queries,
                    "hit_rate": hit_at_k,
                    "threshold": relevance_threshold,
                    "hit_details": hit_details
                }
            ))
        
        return results
    
    def evaluate_retrieval_quality(self, 
                                   queries: List[str], 
                                   k: int = 5) -> EvaluationResult:
        """
        Evaluate retrieval quality metrics
        
        Args:
            queries: List of test queries
            k: Number of sources to retrieve
            
        Returns:
            EvaluationResult with retrieval quality metrics
        """
        print(f"Evaluating retrieval quality with {len(queries)} queries...")
        
        similarity_scores = []
        diversity_scores = []
        
        for query in queries:
            try:
                sources = self.rag_pipeline.faiss_manager.search(query, k=k)
                
                if sources:
                    # Calculate average similarity
                    similarities = [s.get('similarity_score', 0) for s in sources]
                    similarity_scores.extend(similarities)
                    
                    # Calculate diversity (based on unique papers)
                    unique_papers = set(s['paper_id'] for s in sources)
                    diversity = len(unique_papers) / len(sources) if sources else 0
                    diversity_scores.append(diversity)
                
            except Exception as e:
                print(f"Error evaluating retrieval quality for query '{query}': {e}")
                continue
        
        quality_stats = {
            "avg_similarity": statistics.mean(similarity_scores) if similarity_scores else 0,
            "median_similarity": statistics.median(similarity_scores) if similarity_scores else 0,
            "min_similarity": min(similarity_scores) if similarity_scores else 0,
            "max_similarity": max(similarity_scores) if similarity_scores else 0,
            "avg_diversity": statistics.mean(diversity_scores) if diversity_scores else 0,
            "num_queries": len(queries),
            "total_retrieved": len(similarity_scores)
        }
        
        return EvaluationResult(
            metric_name="retrieval_quality",
            value=quality_stats["avg_similarity"],
            details=quality_stats
        )
    
    def run_comprehensive_evaluation(self, 
                                   test_queries: List[str],
                                   output_file: str = "evaluation_results.json") -> Dict[str, Any]:
        """
        Run comprehensive evaluation
        
        Args:
            test_queries: List of test queries
            output_file: File to save results
            
        Returns:
            Dictionary with all evaluation results
        """
        print("Starting comprehensive evaluation...")
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_queries": len(test_queries),
            "evaluations": {}
        }
        
        # Latency evaluation
        print("\n1. Evaluating latency...")
        latency_result = self.evaluate_latency(test_queries)
        results["evaluations"]["latency"] = {
            "metric_name": latency_result.metric_name,
            "value": latency_result.value,
            "details": latency_result.details
        }
        
        # Hit@k evaluation
        print("\n2. Evaluating hit@k...")
        hit_at_k_results = self.evaluate_hit_at_k(test_queries)
        results["evaluations"]["hit_at_k"] = [
            {
                "metric_name": result.metric_name,
                "value": result.value,
                "details": result.details
            }
            for result in hit_at_k_results
        ]
        
        # Retrieval quality evaluation
        print("\n3. Evaluating retrieval quality...")
        quality_result = self.evaluate_retrieval_quality(test_queries)
        results["evaluations"]["retrieval_quality"] = {
            "metric_name": quality_result.metric_name,
            "value": quality_result.value,
            "details": quality_result.details
        }
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEvaluation completed! Results saved to: {output_file}")
        return results


def load_test_queries(file_path: str = "test_queries.json") -> List[str]:
    """Load test queries from file"""
    if Path(file_path).exists():
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get("queries", [])
    else:
        # Default test queries
        return [
            "What are the latest advances in transformer architectures?",
            "How do neural networks learn from data?",
            "What is the difference between supervised and unsupervised learning?",
            "How does attention mechanism work in deep learning?",
            "What are the applications of computer vision in healthcare?",
            "How do reinforcement learning algorithms work?",
            "What is the role of optimization in machine learning?",
            "How do language models understand context?",
            "What are the challenges in natural language processing?",
            "How does deep learning improve image recognition?"
        ]


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QueryGenie Evaluation Script")
    parser.add_argument("--queries", type=str, default="test_queries.json", 
                       help="Path to test queries file")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--latency-only", action="store_true",
                       help="Run only latency evaluation")
    parser.add_argument("--hit-at-k-only", action="store_true",
                       help="Run only hit@k evaluation")
    
    args = parser.parse_args()
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    faiss_manager = FAISSManager()
    
    if not faiss_manager.is_ready():
        print("Error: FAISS manager not ready. Please create an index first.")
        return
    
    rag_pipeline = RAGPipeline(faiss_manager)
    
    # Load test queries
    test_queries = load_test_queries(args.queries)
    print(f"Loaded {len(test_queries)} test queries")
    
    # Initialize evaluator
    evaluator = QueryGenieEvaluator(rag_pipeline)
    
    if args.latency_only:
        # Run only latency evaluation
        result = evaluator.evaluate_latency(test_queries)
        print(f"\nLatency Results:")
        print(f"Average total time: {result.value:.3f}s")
        print(f"Details: {result.details}")
        
    elif args.hit_at_k_only:
        # Run only hit@k evaluation
        results = evaluator.evaluate_hit_at_k(test_queries)
        print(f"\nHit@k Results:")
        for result in results:
            print(f"{result.metric_name}: {result.value:.3f}")
        
    else:
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation(test_queries, args.output)
        
        # Print summary
        print(f"\nEvaluation Summary:")
        print(f"Latency - Average: {results['evaluations']['latency']['value']:.3f}s")
        
        print(f"Hit@k Results:")
        for hit_result in results['evaluations']['hit_at_k']:
            print(f"  {hit_result['metric_name']}: {hit_result['value']:.3f}")
        
        print(f"Retrieval Quality - Average similarity: {results['evaluations']['retrieval_quality']['value']:.3f}")


if __name__ == "__main__":
    main()
