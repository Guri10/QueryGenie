"""
LLM Generator for QueryGenie RAG System
Uses llama-cpp-python for efficient local inference on M2 MacBook
"""

import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
    LLM_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    LLM_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. LLM generation will be disabled.")


@dataclass
class GenerationConfig:
    """Configuration for LLM generation"""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop_sequences: List[str] = None
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = ["\n\nQuestion:", "\n\nContext:", "###"]


class LLMGenerator:
    """
    Local LLM Generator using llama.cpp for efficient inference
    
    Supports quantized models (GGUF format) for M2 MacBook optimization
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0,  # Set to 0 for CPU, increase for GPU
        verbose: bool = False
    ):
        """
        Initialize LLM Generator
        
        Args:
            model_path: Path to local GGUF model file
            model_name: Hugging Face model name (used to download if model_path not provided)
            n_ctx: Context window size
            n_threads: Number of CPU threads to use
            n_gpu_layers: Number of layers to offload to GPU (0 for CPU only)
            verbose: Print debug information
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            )
        
        self.model_path = model_path
        self.model_name = model_name
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.model = None
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model"""
        print(f"Loading LLM model...")
        
        # If no model path provided, download from Hugging Face
        if self.model_path is None:
            self.model_path = self._download_model()
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Please download a GGUF model first."
            )
        
        try:
            # Load model with llama.cpp
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose
            )
            print(f"✅ Model loaded successfully from: {self.model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _download_model(self) -> str:
        """
        Download GGUF model from Hugging Face
        
        Returns:
            Path to downloaded model file
        """
        from huggingface_hub import hf_hub_download
        
        # Model repository and filename mapping
        model_configs = {
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
                "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
            },
            "microsoft/phi-2": {
                "repo_id": "TheBloke/phi-2-GGUF",
                "filename": "phi-2.Q4_K_M.gguf"
            },
            "mistralai/Mistral-7B-Instruct-v0.2": {
                "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            }
        }
        
        # Get model config
        if self.model_name not in model_configs:
            # Default to TinyLlama
            print(f"Unknown model: {self.model_name}, using TinyLlama as default")
            self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        config = model_configs[self.model_name]
        
        print(f"Downloading model: {config['filename']}")
        print(f"From repository: {config['repo_id']}")
        print("This may take a few minutes on first run...")
        
        try:
            model_path = hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename"],
                cache_dir="models/"
            )
            print(f"✅ Model downloaded to: {model_path}")
            return model_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Returns:
            Dictionary with generated text and metadata
        """
        if config is None:
            config = GenerationConfig()
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Generate text
            output = self.model(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                stop=config.stop_sequences,
                echo=False
            )
            
            generation_time = time.time() - start_time
            
            # Extract generated text
            generated_text = output["choices"][0]["text"].strip()
            
            return {
                "text": generated_text,
                "generation_time": generation_time,
                "tokens_generated": output["usage"]["completion_tokens"],
                "tokens_per_second": output["usage"]["completion_tokens"] / generation_time if generation_time > 0 else 0,
                "model": self.model_name
            }
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
    
    def generate_rag_answer(
        self,
        question: str,
        context: str,
        max_answer_length: int = 200
    ) -> Dict[str, Any]:
        """
        Generate answer from question and retrieved context
        
        Args:
            question: User's question
            context: Retrieved context from RAG
            max_answer_length: Maximum length of generated answer
            
        Returns:
            Dictionary with generated answer and metadata
        """
        # Create prompt for RAG
        prompt = self._create_rag_prompt(question, context)
        
        # Configure generation with stricter parameters for better grounding
        config = GenerationConfig(
            max_tokens=max_answer_length,
            temperature=0.3,  # Lower temperature for more focused answers
            top_p=0.85,       # Slightly lower for more deterministic output
            top_k=30,         # Reduced for better focus
            repeat_penalty=1.2,  # Higher to avoid repetition
            stop_sequences=[
                "\n\nQuestion:", 
                "\n\nQUESTION:", 
                "\n\nContext:", 
                "\n\nRESEARCH PAPERS:",
                "###", 
                "\n\n\n",
                "CRITICAL INSTRUCTIONS:"
            ]
        )
        
        # Generate answer - no validation, always return LLM output
        result = self.generate(prompt, config)

        # Post-process: strip any inline reference sections since UI renders sources separately
        try:
            text = result.get("text", "")
            # Remove common reference headers the model might invent
            for header in ["\nREFERENCES:", "\nReferences:", "\nReference:"]:
                idx = text.find(header)
                if idx != -1:
                    text = text[:idx].rstrip()
            # Also trim trailing numbered citation lists if present
            if "[1]" in text and text.strip().endswith("]"):
                # Heuristic: cut off last paragraph if it looks like a citation list
                parts = text.split("\n\n")
                if len(parts) > 1 and any(part.strip().startswith("[1]") for part in parts[-1:]):
                    text = "\n\n".join(parts[:-1]).rstrip()
            result["text"] = text
        except Exception:
            # Best-effort cleanup; ignore failures
            pass

        return result
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """
        Create prompt for RAG answer generation
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a research assistant that answers questions by synthesizing information from the provided research paper abstracts.

INSTRUCTIONS:
- Base your answer on the research papers provided below.
- Synthesize the information into a coherent, informative response.
- Do NOT include references, citations, or a REFERENCES section in your answer. The UI will show sources separately.
- If the papers don't contain relevant information, say "The provided papers don't contain information about this topic".
- Focus on the technical/academic content from the papers.
- Write in a clear, informative style that helps the user understand the topic.

RESEARCH PAPERS:
{context}

QUESTION: {question}

ANSWER (synthesizing information from the research papers above; no citations or references in the text): """
        
        return prompt
    
    def _validate_answer_grounding(self, answer: str, context: str, question: str) -> Dict[str, Any]:
        """
        Validate that the generated answer is properly grounded in the provided context
        
        Args:
            answer: Generated answer text
            context: Retrieved context from papers
            question: Original question
            
        Returns:
            Dictionary with validation results
        """
        answer_lower = answer.lower()
        context_lower = context.lower()
        question_lower = question.lower()
        
        # Check 1: Answer should contain citation numbers OR reference to papers
        has_citations = any(f'[{i}]' in answer for i in range(1, 10))
        references_papers = any(term in answer_lower for term in [
            'paper', 'papers', 'research', 'study', 'according to', 'based on',
            'the research', 'the study', 'the paper', 'this paper'
        ])
        
        # Check 2: Answer should contain key terms from context
        # Extract key technical terms from context (words that appear multiple times)
        context_words = context_lower.split()
        word_counts = {}
        for word in context_words:
            if len(word) > 4:  # Only consider longer words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get frequent technical terms from context
        technical_terms = [word for word, count in word_counts.items() if count >= 2 and len(word) > 4]
        
        # Check if answer contains these technical terms
        context_term_matches = sum(1 for term in technical_terms if term in answer_lower)
        context_term_ratio = context_term_matches / max(len(technical_terms), 1)
        
        # Check 3: Answer should NOT contain completely unrelated terms
        # Check for common hallucination patterns
        unrelated_terms = [
            'law libraries', 'museums', 'legal history', 'legal profession',
            'limited liability partnership', 'business entity', 'partnership',
            'law librarian', 'librarian', 'legal materials', 'legal collection',
            'cooking', 'recipes', 'food', 'sports', 'entertainment'
        ]
        
        has_unrelated = any(term in answer_lower for term in unrelated_terms)
        
        # Check 4: Answer should be related to the question topic
        # Extract key terms from question (including shorter terms)
        question_terms = [word for word in question_lower.split() if len(word) > 1]
        question_term_matches = sum(1 for term in question_terms if term in answer_lower)
        question_relevance = question_term_matches / max(len(question_terms), 1)
        
        # Check 5: Answer length should be reasonable
        reasonable_length = 20 <= len(answer.strip()) <= 1000
        
        # Overall validation - more lenient but still prevents hallucination
        is_grounded = (
            (has_citations or references_papers or context_term_ratio >= 0.1) and  # Must reference papers OR use context terms
            context_term_ratio >= 0.05 and  # Must use some context terms
            not has_unrelated and  # Must not have unrelated content
            question_relevance >= 0.1 and  # Must be relevant to question (lowered from 0.2)
            reasonable_length  # Must be reasonable length
        )
        
        return {
            'is_grounded': is_grounded,
            'has_citations': has_citations,
            'references_papers': references_papers,
            'context_term_ratio': context_term_ratio,
            'has_unrelated': has_unrelated,
            'question_relevance': question_relevance,
            'reasonable_length': reasonable_length,
            'technical_terms_found': context_term_matches,
            'total_technical_terms': len(technical_terms)
        }
    
    def is_ready(self) -> bool:
        """Check if generator is ready"""
        return self.model is not None


def main():
    """Test the LLM generator"""
    print("Testing LLM Generator...")
    
    # Initialize generator (will download model on first run)
    generator = LLMGenerator(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_threads=4,
        verbose=False
    )
    
    # Test simple generation
    print("\nTest 1: Simple generation")
    result = generator.generate("What is machine learning?")
    print(f"Generated text: {result['text']}")
    print(f"Generation time: {result['generation_time']:.2f}s")
    print(f"Tokens/second: {result['tokens_per_second']:.2f}")
    
    # Test RAG answer generation
    print("\nTest 2: RAG answer generation")
    question = "What are transformer architectures?"
    context = """
    [1] Title: Attention Is All You Need
    Abstract: The dominant sequence transduction models are based on complex recurrent 
    or convolutional neural networks that include an encoder and a decoder. The best 
    performing models also connect the encoder and decoder through an attention mechanism. 
    We propose a new simple network architecture, the Transformer, based solely on 
    attention mechanisms, dispensing with recurrence and convolutions entirely.
    """
    
    result = generator.generate_rag_answer(question, context, max_answer_length=150)
    print(f"Question: {question}")
    print(f"Answer: {result['text']}")
    print(f"Generation time: {result['generation_time']:.2f}s")
    
    print("\n✅ LLM Generator test completed!")


if __name__ == "__main__":
    main()

