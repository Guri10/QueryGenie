"""
FAISS Index Optimizer for QueryGenie
Provides faster index types for large-scale retrieval
"""

import faiss
import numpy as np
from typing import Tuple, Optional


class FAISSOptimizer:
    """
    Optimizes FAISS indexes for better performance
    
    Strategies:
    1. IVF (Inverted File) index for large datasets
    2. PQ (Product Quantization) for memory efficiency
    3. HNSW for ultra-fast approximate search
    """
    
    @staticmethod
    def create_optimized_index(
        embeddings: np.ndarray,
        index_type: str = "auto",
        nlist: int = 100,
        nprobe: int = 10
    ) -> faiss.Index:
        """
        Create optimized FAISS index
        
        Args:
            embeddings: Embedding vectors (n_samples, n_dimensions)
            index_type: Type of index ('flat', 'ivf', 'hnsw', 'auto')
            nlist: Number of clusters for IVF (default: 100)
            nprobe: Number of clusters to search (default: 10)
            
        Returns:
            Optimized FAISS index
        """
        n_samples, dimension = embeddings.shape
        
        # Auto-select index type based on dataset size
        if index_type == "auto":
            if n_samples < 1000:
                index_type = "flat"  # Exact search for small datasets
            elif n_samples < 10000:
                index_type = "ivf"   # IVF for medium datasets
            else:
                index_type = "hnsw"  # HNSW for large datasets
        
        print(f"Creating {index_type.upper()} index for {n_samples} vectors...")
        
        if index_type == "flat":
            # Exact search (current approach)
            index = faiss.IndexFlatIP(dimension)
            
        elif index_type == "ivf":
            # IVF (Inverted File) - faster approximate search
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Train the index
            print(f"Training IVF index with {nlist} clusters...")
            index.train(embeddings.astype('float32'))
            index.nprobe = nprobe  # Search nprobe clusters
            
        elif index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World) - very fast
            M = 32  # Number of connections per layer
            index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        return index
    
    @staticmethod
    def benchmark_index_types(
        embeddings: np.ndarray,
        query_vectors: np.ndarray,
        k: int = 5
    ) -> dict:
        """
        Benchmark different index types
        
        Args:
            embeddings: Training embeddings
            query_vectors: Query vectors for testing
            k: Number of results to retrieve
            
        Returns:
            Performance comparison dict
        """
        import time
        
        results = {}
        
        for index_type in ["flat", "ivf", "hnsw"]:
            try:
                # Create index
                index = FAISSOptimizer.create_optimized_index(
                    embeddings,
                    index_type=index_type
                )
                
                # Normalize and add vectors
                normalized = embeddings.copy()
                faiss.normalize_L2(normalized)
                index.add(normalized.astype('float32'))
                
                # Benchmark search
                times = []
                for query in query_vectors:
                    q = query.reshape(1, -1).astype('float32')
                    faiss.normalize_L2(q)
                    
                    start = time.time()
                    scores, indices = index.search(q, k)
                    times.append(time.time() - start)
                
                results[index_type] = {
                    "avg_time": np.mean(times),
                    "std_time": np.std(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times)
                }
                
            except Exception as e:
                results[index_type] = {"error": str(e)}
        
        return results
    
    @staticmethod
    def optimize_existing_index(
        index_path: str,
        embeddings: np.ndarray,
        target_type: str = "ivf"
    ) -> Tuple[faiss.Index, str]:
        """
        Convert existing index to optimized version
        
        Args:
            index_path: Path to existing index
            embeddings: Original embeddings
            target_type: Target index type
            
        Returns:
            Tuple of (new_index, new_index_path)
        """
        # Create optimized index
        new_index = FAISSOptimizer.create_optimized_index(
            embeddings,
            index_type=target_type
        )
        
        # Normalize and add vectors
        normalized = embeddings.copy()
        faiss.normalize_L2(normalized)
        new_index.add(normalized.astype('float32'))
        
        # Save with new name
        new_path = index_path.replace('.faiss', f'_{target_type}.faiss')
        faiss.write_index(new_index, new_path)
        
        return new_index, new_path


def main():
    """Test FAISS optimizer"""
    print("Testing FAISS Optimizer...")
    
    # Create sample data
    n_samples = 1000
    dimension = 384
    embeddings = np.random.randn(n_samples, dimension).astype('float32')
    queries = np.random.randn(10, dimension).astype('float32')
    
    # Benchmark
    results = FAISSOptimizer.benchmark_index_types(embeddings, queries, k=5)
    
    print("\nBenchmark Results:")
    print("=" * 60)
    for index_type, metrics in results.items():
        if "error" in metrics:
            print(f"{index_type.upper()}: Error - {metrics['error']}")
        else:
            print(f"{index_type.upper()}:")
            print(f"  Average time: {metrics['avg_time']*1000:.2f}ms")
            print(f"  Min time:     {metrics['min_time']*1000:.2f}ms")
            print(f"  Max time:     {metrics['max_time']*1000:.2f}ms")
    
    print("\n✅ FAISS Optimizer test completed!")


if __name__ == "__main__":
    main()

