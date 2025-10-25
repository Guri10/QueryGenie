"""
RAG Pipeline
Retrieval-Augmented Generation with local models
"""

from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass

try:
    from llm_generator import LLMGenerator, GenerationConfig
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


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
    
    def __init__(
        self, 
        faiss_manager, 
        use_llm: bool = False,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model_path: Optional[str] = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            faiss_manager: FAISSManager instance
            use_llm: Whether to use LLM for text generation
            model_name: Name of the LLM model to use
            model_path: Path to local GGUF model file
        """
        self.faiss_manager = faiss_manager
        self.use_llm = use_llm
        self.generator = None
        
        if use_llm:
            if not LLM_AVAILABLE:
                print("⚠️ Warning: LLM dependencies not available. Falling back to retrieval-only mode.")
                print("   Install with: pip install llama-cpp-python huggingface-hub")
                self.use_llm = False
            else:
                try:
                    print(f"Initializing LLM generator with {model_name}...")
                    self.generator = LLMGenerator(
                        model_path=model_path,
                        model_name=model_name,
                        n_threads=4,
                        verbose=False
                    )
                    print("✅ RAG pipeline running with LLM generation")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to load LLM: {e}")
                    print("   Falling back to retrieval-only mode")
                    self.use_llm = False
                    self.generator = None
        
        if not self.use_llm:
            print("RAG pipeline running in retrieval-only mode")
        
    def format_context(self, sources: List[Dict[str, Any]], max_context_length: int = 5000) -> str:
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
              max_context_length: int = 5000,
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
        sources = self.faiss_manager.search(question, k=k, return_scores=True)
        retrieval_time = time.time() - retrieval_start
        
        if not sources:
            model_name = self.generator.model_name if self.generator else "retrieval-only"
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                retrieval_time=retrieval_time,
                generation_time=0.0,
                total_time=time.time() - start_time,
                model_used=model_name
            )
        
        # Step 2: Format context
        context = self.format_context(sources, max_context_length)
        
        # Step 3: Generate answer
        generation_start = time.time()
        
        if self.use_llm and self.generator:
            # LLM-powered generation
            try:
                result = self.generator.generate_rag_answer(
                    question=question,
                    context=context,
                    max_answer_length=max_answer_length
                )
                answer = result['text']
                generation_time = result['generation_time']
                model_name = result['model']
            except Exception as e:
                print(f"⚠️ LLM generation failed: {e}. Falling back to retrieval-only.")
                answer = f"Based on the research papers, here's what I found:\n\n{context[:max_answer_length]}"
                generation_time = time.time() - generation_start
                model_name = "retrieval-only (fallback)"
        else:
            # Retrieval-only mode
            answer = f"Based on the research papers, here's what I found:\n\n{context[:max_answer_length]}"
            generation_time = time.time() - generation_start
            model_name = "retrieval-only"
        
        # Step 4: Format citations
        citations = self.format_citations(sources)
        
        return RAGResponse(
            answer=answer,
            sources=citations,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=time.time() - start_time,
            model_used=model_name
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        faiss_stats = self.faiss_manager.get_stats()
        
        generator_info = {
            "enabled": self.use_llm,
            "model": self.generator.model_name if self.generator else None,
            "ready": self.generator.is_ready() if self.generator else False
        }
        
        return {
            "faiss_stats": faiss_stats,
            "generator_info": generator_info,
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
