"""
Test script for LLM generation in QueryGenie
"""

import sys

def test_imports():
    """Test if dependencies are installed"""
    print("Testing imports...")
    
    try:
        import llama_cpp
        print("✅ llama-cpp-python installed")
        llm_available = True
    except ImportError:
        print("❌ llama-cpp-python NOT installed")
        print("   Install with: pip install llama-cpp-python")
        llm_available = False
    
    try:
        import huggingface_hub
        print("✅ huggingface-hub installed")
        hf_available = True
    except ImportError:
        print("❌ huggingface-hub NOT installed")
        print("   Install with: pip install huggingface-hub")
        hf_available = False
    
    return llm_available and hf_available


def test_llm_generator():
    """Test LLM generator module"""
    print("\nTesting LLM generator module...")
    
    try:
        from src.llm_generator import LLMGenerator, GenerationConfig, LLAMA_CPP_AVAILABLE
        print("✅ LLM generator module imported successfully")
        
        if not LLAMA_CPP_AVAILABLE:
            print("⚠️  llama-cpp-python not available")
            return False
        
        print("✅ LLM dependencies available")
        return True
        
    except Exception as e:
        print(f"❌ Error importing LLM generator: {e}")
        return False


def test_rag_pipeline():
    """Test RAG pipeline with LLM support"""
    print("\nTesting RAG pipeline with LLM support...")
    
    try:
        from src.rag_pipeline import RAGPipeline, LLM_AVAILABLE
        print("✅ RAG pipeline imported successfully")
        
        if LLM_AVAILABLE:
            print("✅ LLM support is available in RAG pipeline")
        else:
            print("⚠️  LLM support not available (dependencies missing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing RAG pipeline: {e}")
        return False


def test_faiss_retrieval():
    """Test retrieval-only mode (no LLM)"""
    print("\nTesting retrieval-only mode...")
    
    try:
        from src.faiss_manager import FAISSManager
        from src.rag_pipeline import RAGPipeline
        
        faiss_manager = FAISSManager()
        
        if not faiss_manager.is_ready():
            print("⚠️  FAISS index not found. Skipping retrieval test.")
            print("   Run: python src/preprocessing.py")
            return False
        
        # Test retrieval-only mode
        rag = RAGPipeline(faiss_manager, use_llm=False)
        response = rag.query("What is machine learning?", k=3)
        
        print(f"✅ Retrieval-only mode working")
        print(f"   Query time: {response.total_time:.3f}s")
        print(f"   Model: {response.model_used}")
        print(f"   Sources retrieved: {len(response.sources)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing retrieval: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("QueryGenie LLM Generation Test Suite")
    print("=" * 60)
    
    results = {
        "imports": test_imports(),
        "llm_module": test_llm_generator(),
        "rag_pipeline": test_rag_pipeline(),
        "retrieval": test_faiss_retrieval()
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Install LLM dependencies:")
        print("   pip install llama-cpp-python huggingface-hub")
        print("2. Enable LLM generation:")
        print("   USE_LLM=true python src/api.py")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        print("\nTo install LLM dependencies:")
        print("   pip install llama-cpp-python huggingface-hub")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

