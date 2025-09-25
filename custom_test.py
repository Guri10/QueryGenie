#!/usr/bin/env python3
"""
Custom Testing Script for QueryGenie RAG System
Allows you to test with your own questions and parameters
"""

import requests
import json
import time
import sys
from typing import List, Dict, Any

class QueryGenieTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self) -> bool:
        """Test if the system is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ System Status: {data['status']}")
                print(f"✅ Pipeline Ready: {data['pipeline_ready']}")
                print(f"✅ Total Papers: {data['index_stats']['total_papers']}")
                print(f"✅ Total Chunks: {data['index_stats']['total_chunks']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def ask_question(self, 
                    question: str, 
                    k: int = 5,
                    max_context_length: int = 1000,
                    max_answer_length: int = 200,
                    show_details: bool = True) -> Dict[str, Any]:
        """Ask a custom question to the RAG system"""
        
        print(f"\n🔍 Question: {question}")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/ask",
                json={
                    "question": question,
                    "k": k,
                    "max_context_length": max_context_length,
                    "max_answer_length": max_answer_length
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Answer: {data['answer']}")
                print(f"⏱️  Response Time: {data['total_time']:.3f}s")
                print(f"🔍 Retrieval Time: {data['retrieval_time']:.3f}s")
                print(f"🤖 Generation Time: {data['generation_time']:.3f}s")
                print(f"📊 Model Used: {data['model_used']}")
                
                if show_details and data['sources']:
                    print(f"\n📚 Sources ({len(data['sources'])}):")
                    for i, source in enumerate(data['sources'], 1):
                        print(f"  [{i}] {source['paper_title']}")
                        print(f"      Authors: {', '.join(source['authors'][:3])}")
                        print(f"      Similarity: {source['similarity_score']:.3f}")
                        print(f"      Preview: {source['text_preview'][:150]}...")
                        print()
                
                return data
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {}
    
    def batch_test(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Test multiple questions in batch"""
        print(f"\n🧪 Batch Testing {len(questions)} questions...")
        print("=" * 80)
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Testing question...")
            result = self.ask_question(question, **kwargs)
            results.append(result)
            time.sleep(0.5)  # Small delay between requests
        
        return results
    
    def performance_test(self, question: str, num_runs: int = 5) -> Dict[str, Any]:
        """Test performance with multiple runs"""
        print(f"\n⚡ Performance Test: {num_runs} runs")
        print("=" * 80)
        
        times = []
        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs}...")
            result = self.ask_question(question, show_details=False)
            if result:
                times.append(result['total_time'])
            time.sleep(0.2)
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\n📊 Performance Results:")
            print(f"   Average Time: {avg_time:.3f}s")
            print(f"   Min Time: {min_time:.3f}s")
            print(f"   Max Time: {max_time:.3f}s")
            print(f"   Std Dev: {(sum((t - avg_time)**2 for t in times) / len(times))**0.5:.3f}s")
            
            return {
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "times": times
            }
        
        return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            response = self.session.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Metrics failed: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ Metrics error: {e}")
            return {}

def interactive_mode():
    """Interactive testing mode"""
    tester = QueryGenieTester()
    
    print("🚀 QueryGenie Custom Testing - Interactive Mode")
    print("=" * 60)
    
    # Check health first
    if not tester.test_health():
        print("❌ System not healthy. Please check if the server is running.")
        return
    
    print("\n💡 Interactive Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'batch' to test multiple questions")
    print("  - Type 'perf' for performance testing")
    print("  - Type 'metrics' to see system metrics")
    print("  - Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n🔍 Enter your question (or command): ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'metrics':
                metrics = tester.get_metrics()
                if metrics:
                    print(f"📊 System Metrics:")
                    print(f"   Total Queries: {metrics['total_queries']}")
                    print(f"   Avg Response Time: {metrics['average_total_time']:.3f}s")
                    print(f"   Avg Retrieval Time: {metrics['average_retrieval_time']:.3f}s")
            elif user_input.lower() == 'batch':
                questions = [
                    "What are the latest advances in deep learning?",
                    "How do neural networks work?",
                    "What is machine learning?",
                    "Tell me about computer vision",
                    "What are the challenges in AI?"
                ]
                tester.batch_test(questions)
            elif user_input.lower() == 'perf':
                question = input("Enter question for performance test: ").strip()
                if question:
                    tester.performance_test(question)
            elif user_input:
                tester.ask_question(user_input)
            else:
                print("Please enter a question or command.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function with different testing modes"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "interactive":
            interactive_mode()
        elif mode == "batch":
            # Batch testing with predefined questions
            tester = QueryGenieTester()
            if tester.test_health():
                questions = [
                    "What are the latest advances in machine learning?",
                    "How do transformer architectures work?",
                    "What is reinforcement learning?",
                    "Tell me about computer vision applications",
                    "What are the challenges in natural language processing?",
                    "What is deep learning?",
                    "How do neural networks learn?",
                    "What are the applications of AI?",
                    "What is computer vision?",
                    "How does attention mechanism work?"
                ]
                tester.batch_test(questions)
        elif mode == "perf":
            # Performance testing
            tester = QueryGenieTester()
            if tester.test_health():
                question = "What are the latest advances in machine learning?"
                tester.performance_test(question, num_runs=10)
        else:
            print("Usage: python custom_test.py [interactive|batch|perf]")
    else:
        # Default: interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()
