"""
RAG Pipeline
Retrieval-Augmented Generation with local models
"""

from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass


@dataclass
class RAGResponse:
    """Response from RAG pipeline"""
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float
    total_time: float
    model_used: str




class RAGPipeline:
    """Complete RAG pipeline with retrieval and generation"""
    
    def __init__(self, faiss_manager):
        """
        Initialize RAG pipeline (retrieval-only mode)
        
        Args:
            faiss_manager: FAISSManager instance
        """
        self.faiss_manager = faiss_manager
        print("RAG pipeline running in retrieval-only mode")
        
    def format_context(self, sources: List[Dict[str, Any]], max_context_length: int = 1000) -> str:
        """
        Format retrieved sources into context
        
        Args:
            sources: List of retrieved sources
            max_context_length: Maximum length of context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, source in enumerate(sources):
            # Format source with citation
            citation = f"[{i+1}]"
            source_text = f"{citation} {source['text']}"
            
            # Check if adding this source would exceed max length
            if current_length + len(source_text) > max_context_length:
                break
                
            context_parts.append(source_text)
            current_length += len(source_text)
        
        return "\n\n".join(context_parts)
    
    def format_citations(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format sources for citation display
        
        Args:
            sources: List of retrieved sources
            
        Returns:
            Formatted citations
        """
        citations = []
        for i, source in enumerate(sources):
            citation = {
                'number': i + 1,
                'paper_title': source['paper_title'],
                'authors': source['authors'],
                'paper_id': source['paper_id'],
                'similarity_score': source.get('similarity_score', 0.0),
                'text_preview': source['text'][:200] + "..." if len(source['text']) > 200 else source['text']
            }
            citations.append(citation)
        
        return citations
    
    def query(self, 
              question: str, 
              k: int = 5,
              max_context_length: int = 1000,
              max_answer_length: int = 200) -> RAGResponse:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User question
            k: Number of sources to retrieve
            max_context_length: Maximum context length
            max_answer_length: Maximum answer length
            
        Returns:
            RAGResponse with answer and sources
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant sources
        retrieval_start = time.time()
        sources = self.faiss_manager.search(question, k=k)
        retrieval_time = time.time() - retrieval_start
        
        if not sources:
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                retrieval_time=retrieval_time,
                generation_time=0.0,
                total_time=time.time() - start_time,
                model_used=self.generator.model_name
            )
        
        # Step 2: Format context
        context = self.format_context(sources, max_context_length)
        
        # Step 3: Create prompt
        prompt = f"""Based on the following research papers, please answer the question: {question}

Context:
{context}

Answer:"""
        
        # Step 4: Generate answer (retrieval-only mode)
        generation_start = time.time()
        answer = f"Based on the research papers, here's what I found:\n\n{context[:max_answer_length]}"
        generation_time = time.time() - generation_start
        
        # Step 5: Format citations
        citations = self.format_citations(sources)
        
        return RAGResponse(
            answer=answer,
            sources=citations,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=time.time() - start_time,
            model_used="retrieval-only"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        faiss_stats = self.faiss_manager.get_stats()
        
        return {
            "faiss_stats": faiss_stats,
            "generator_model": None,
            "generator_device": None,
            "pipeline_ready": self.faiss_manager.is_ready()
        }


def main():
    """Test RAG pipeline"""
    from faiss_manager import FAISSManager
    
    # Initialize components
    faiss_manager = FAISSManager()
    
    if not faiss_manager.is_ready():
        print("FAISS manager not ready. Please create an index first.")
        return
    
    # Initialize RAG pipeline
    rag = RAGPipeline(faiss_manager)
    
    # Test query
    question = "What are the latest advances in transformer architectures?"
    print(f"Question: {question}")
    
    response = rag.query(question, k=3)
    
    print(f"\nAnswer: {response.answer}")
    print(f"\nRetrieval time: {response.retrieval_time:.3f}s")
    print(f"Generation time: {response.generation_time:.3f}s")
    print(f"Total time: {response.total_time:.3f}s")
    
    print(f"\nSources:")
    for source in response.sources:
        print(f"[{source['number']}] {source['paper_title']}")
        print(f"    Authors: {', '.join(source['authors'][:3])}")
        print(f"    Similarity: {source['similarity_score']:.3f}")
        print(f"    Preview: {source['text_preview']}")
        print()


if __name__ == "__main__":
    main()
