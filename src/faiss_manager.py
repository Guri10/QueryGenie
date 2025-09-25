"""
FAISS Index Manager
Handles loading, searching, and managing FAISS indices
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


class FAISSManager:
    def __init__(self, 
                 index_path: str = "data/faiss_index",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize FAISS manager
        
        Args:
            index_path: Path to FAISS index files
            embedding_model: Hugging Face model for embeddings
        """
        self.index_path = index_path
        self.embedding_model_name = embedding_model
        
        # Load components
        self.index = None
        self.chunks_metadata = []
        self.embedder = None
        
        self._load_components()
    
    def _load_components(self):
        """Load FAISS index, chunks metadata, and embedding model"""
        try:
            # Load FAISS index
            if os.path.exists(f"{self.index_path}.faiss"):
                print(f"Loading FAISS index from: {self.index_path}.faiss")
                self.index = faiss.read_index(f"{self.index_path}.faiss")
                print(f"Index loaded with {self.index.ntotal} vectors")
            else:
                print(f"FAISS index not found at: {self.index_path}.faiss")
                return
            
            # Load chunks metadata
            if os.path.exists(f"{self.index_path}_chunks.json"):
                print(f"Loading chunks metadata from: {self.index_path}_chunks.json")
                with open(f"{self.index_path}_chunks.json", 'r', encoding='utf-8') as f:
                    self.chunks_metadata = json.load(f)
                print(f"Loaded {len(self.chunks_metadata)} chunks metadata")
            else:
                print(f"Chunks metadata not found at: {self.index_path}_chunks.json")
                return
            
            # Load embedding model
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedder = SentenceTransformer(self.embedding_model_name)
            print("Embedding model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading components: {e}")
            self.index = None
            self.chunks_metadata = []
            self.embedder = None
    
    def is_ready(self) -> bool:
        """Check if all components are loaded and ready"""
        return (self.index is not None and 
                len(self.chunks_metadata) > 0 and 
                self.embedder is not None)
    
    def search(self, 
               query: str, 
               k: int = 5,
               return_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query: Search query
            k: Number of results to return
            return_scores: Whether to return similarity scores
            
        Returns:
            List of search results with metadata
        """
        if not self.is_ready():
            raise RuntimeError("FAISS manager not ready. Please check if index and metadata are loaded.")
        
        # Encode query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks_metadata):
                chunk_data = self.chunks_metadata[idx]
                result = {
                    'rank': i + 1,
                    'chunk_id': chunk_data['chunk_id'],
                    'text': chunk_data['text'],
                    'paper_id': chunk_data['paper_id'],
                    'paper_title': chunk_data['paper_title'],
                    'authors': chunk_data['authors'],
                    'chunk_index': chunk_data['chunk_index'],
                    'total_chunks': chunk_data['total_chunks'],
                    'metadata': chunk_data['metadata']
                }
                
                if return_scores:
                    result['similarity_score'] = float(score)
                
                results.append(result)
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """Get a specific chunk by its ID"""
        for chunk in self.chunks_metadata:
            if chunk['chunk_id'] == chunk_id:
                return chunk
        return None
    
    def get_paper_chunks(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific paper"""
        return [chunk for chunk in self.chunks_metadata if chunk['paper_id'] == paper_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        if not self.is_ready():
            return {"error": "Index not ready"}
        
        # Count unique papers
        unique_papers = set(chunk['paper_id'] for chunk in self.chunks_metadata)
        
        # Count chunks per paper
        chunks_per_paper = {}
        for chunk in self.chunks_metadata:
            paper_id = chunk['paper_id']
            chunks_per_paper[paper_id] = chunks_per_paper.get(paper_id, 0) + 1
        
        return {
            "total_chunks": len(self.chunks_metadata),
            "total_papers": len(unique_papers),
            "index_vectors": self.index.ntotal,
            "embedding_dimension": self.index.d,
            "chunks_per_paper": chunks_per_paper,
            "embedding_model": self.embedding_model_name
        }
    
    def reload_index(self):
        """Reload the index and metadata"""
        print("Reloading FAISS index...")
        self._load_components()
        
        if self.is_ready():
            print("Index reloaded successfully!")
        else:
            print("Failed to reload index. Please check file paths.")


def main():
    """Test FAISS manager"""
    manager = FAISSManager()
    
    if manager.is_ready():
        print("FAISS Manager is ready!")
        
        # Test search
        query = "machine learning neural networks"
        results = manager.search(query, k=3)
        
        print(f"\nSearch results for: '{query}'")
        for result in results:
            print(f"\nRank {result['rank']}: {result['paper_title']}")
            print(f"Similarity: {result['similarity_score']:.4f}")
            print(f"Text preview: {result['text'][:200]}...")
        
        # Show stats
        stats = manager.get_stats()
        print(f"\nIndex Statistics:")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total papers: {stats['total_papers']}")
        print(f"Index vectors: {stats['index_vectors']}")
        
    else:
        print("FAISS Manager not ready. Please create an index first.")


if __name__ == "__main__":
    main()
