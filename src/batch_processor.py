"""
Batch Query Processor for QueryGenie
Processes multiple queries efficiently in batches
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class BatchQuery:
    """Single query in a batch"""
    query_id: str
    question: str
    k: int = 5


@dataclass
class BatchResult:
    """Result for a batch of queries"""
    query_id: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float


class BatchQueryProcessor:
    """
    Process multiple queries in batches for better performance
    
    Benefits:
    1. Batch embedding generation (faster than one-by-one)
    2. Parallel FAISS searches
    3. Reduced overhead
    """
    
    def __init__(self, faiss_manager, rag_pipeline):
        """
        Initialize batch processor
        
        Args:
            faiss_manager: FAISSManager instance
            rag_pipeline: RAGPipeline instance
        """
        self.faiss_manager = faiss_manager
        self.rag_pipeline = rag_pipeline
    
    def process_batch(
        self,
        queries: List[BatchQuery],
        max_batch_size: int = 32
    ) -> List[BatchResult]:
        """
        Process multiple queries in batches
        
        Args:
            queries: List of queries to process
            max_batch_size: Maximum batch size for embedding
            
        Returns:
            List of results
        """
        results = []
        
        # Process in chunks of max_batch_size
        for i in range(0, len(queries), max_batch_size):
            batch = queries[i:i + max_batch_size]
            batch_results = self._process_single_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_single_batch(self, batch: List[BatchQuery]) -> List[BatchResult]:
        """Process a single batch of queries"""
        start_time = time.time()
        
        # Extract questions
        questions = [q.question for q in batch]
        
        # Batch encode all queries at once (faster!)
        query_embeddings = self.faiss_manager.embedder.encode(
            questions,
            convert_to_numpy=True,
            batch_size=len(questions)
        )
        
        embedding_time = time.time() - start_time
        
        # Process each query with pre-computed embedding
        results = []
        for i, query in enumerate(batch):
            retrieval_start = time.time()
            
            # Use pre-computed embedding for search
            import faiss
            query_embedding = query_embeddings[i:i+1]
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.faiss_manager.index.search(
                query_embedding.astype('float32'),
                query.k
            )
            
            # Get sources
            sources = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.faiss_manager.chunks_metadata):
                    chunk_data = self.faiss_manager.chunks_metadata[idx]
                    sources.append({
                        'rank': rank + 1,
                        'paper_title': chunk_data['paper_title'],
                        'authors': chunk_data['authors'],
                        'text': chunk_data['text'],
                        'similarity_score': float(score)
                    })
            
            retrieval_time = time.time() - retrieval_start
            
            # Format answer
            context = self.rag_pipeline.format_context(sources, max_context_length=1000)
            answer = f"Based on the research papers, here's what I found:\n\n{context[:200]}"
            
            results.append(BatchResult(
                query_id=query.query_id,
                answer=answer,
                sources=sources,
                retrieval_time=retrieval_time,
                generation_time=0.0
            ))
        
        return results


def main():
    """Test batch processor"""
    from src.faiss_manager import FAISSManager
    from src.rag_pipeline import RAGPipeline
    
    print("Testing Batch Query Processor...")
    
    # Initialize
    fm = FAISSManager()
    rag = RAGPipeline(fm, use_llm=False)
    batch_processor = BatchQueryProcessor(fm, rag)
    
    # Create batch queries
    queries = [
        BatchQuery("q1", "What is machine learning?", k=3),
        BatchQuery("q2", "How do transformers work?", k=3),
        BatchQuery("q3", "What is deep learning?", k=3),
    ]
    
    # Process batch
    start = time.time()
    results = batch_processor.process_batch(queries)
    batch_time = time.time() - start
    
    print(f"\nProcessed {len(queries)} queries in {batch_time:.3f}s")
    print(f"Average time per query: {batch_time/len(queries):.3f}s")
    
    print("\n✅ Batch processor test completed!")


if __name__ == "__main__":
    main()

