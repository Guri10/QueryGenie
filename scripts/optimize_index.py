"""
Script to optimize existing FAISS index
Migrates from flat index to IVF or HNSW for better performance
"""

import sys
import os
import argparse
import numpy as np
import faiss

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from faiss_optimizer import FAISSOptimizer
from preprocessing import DocumentProcessor


def load_existing_embeddings(index_path: str) -> np.ndarray:
    """
    Extract embeddings from existing FAISS index
    
    Args:
        index_path: Path to existing index
        
    Returns:
        Embeddings array
    """
    print(f"Loading existing index: {index_path}")
    index = faiss.read_index(index_path)
    
    # Extract vectors from index
    n_vectors = index.ntotal
    dimension = index.d
    
    print(f"Index contains {n_vectors} vectors of dimension {dimension}")
    
    # Reconstruct all vectors
    embeddings = np.zeros((n_vectors, dimension), dtype='float32')
    for i in range(n_vectors):
        embeddings[i] = index.reconstruct(i)
    
    return embeddings


def optimize_index(
    index_path: str = "data/faiss_index.faiss",
    target_type: str = "ivf",
    backup: bool = True
):
    """
    Optimize existing FAISS index
    
    Args:
        index_path: Path to existing index
        target_type: Target index type ('ivf' or 'hnsw')
        backup: Whether to backup original index
    """
    print("=" * 60)
    print("FAISS Index Optimization Tool")
    print("=" * 60)
    
    # Check if index exists
    if not os.path.exists(index_path):
        print(f"Error: Index not found at {index_path}")
        return
    
    # Load existing embeddings
    try:
        embeddings = load_existing_embeddings(index_path)
    except Exception as e:
        print(f"Error loading index: {e}")
        print("\nNote: If using IVF/HNSW index, vectors cannot be reconstructed.")
        print("Please re-run preprocessing with --index-type flag instead.")
        return
    
    # Backup original index
    if backup:
        backup_path = index_path.replace('.faiss', '_backup.faiss')
        print(f"\nCreating backup: {backup_path}")
        import shutil
        shutil.copy2(index_path, backup_path)
        print("✅ Backup created")
    
    # Create optimized index
    print(f"\nCreating optimized {target_type.upper()} index...")
    new_index, new_path = FAISSOptimizer.optimize_existing_index(
        index_path,
        embeddings,
        target_type
    )
    
    print(f"✅ Optimized index saved to: {new_path}")
    
    # Benchmark
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    # Create test queries
    n_test = min(10, embeddings.shape[0])
    test_queries = embeddings[np.random.choice(embeddings.shape[0], n_test, replace=False)]
    
    results = FAISSOptimizer.benchmark_index_types(
        embeddings,
        test_queries,
        k=5
    )
    
    for index_type, metrics in results.items():
        if "error" not in metrics:
            print(f"\n{index_type.upper()}:")
            print(f"  Average time: {metrics['avg_time']*1000:.2f}ms")
            print(f"  Min time:     {metrics['min_time']*1000:.2f}ms")
            print(f"  Max time:     {metrics['max_time']*1000:.2f}ms")
    
    # Calculate speedup
    if "flat" in results and target_type in results:
        if "error" not in results["flat"] and "error" not in results[target_type]:
            speedup = results["flat"]["avg_time"] / results[target_type]["avg_time"]
            print(f"\n🚀 Speedup: {speedup:.1f}x faster!")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print(f"1. Test the new index:")
    print(f"   python -c \"from src.faiss_manager import FAISSManager; fm = FAISSManager(index_path='{new_path.replace('.faiss', '')}'); print('Index loaded:', fm.is_ready())\"")
    print(f"\n2. If satisfied, replace the original:")
    print(f"   mv {new_path} {index_path}")
    print(f"\n3. Update faiss_manager.py to use the new index")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Optimize FAISS index for better performance")
    parser.add_argument(
        "--index-path",
        type=str,
        default="data/faiss_index.faiss",
        help="Path to existing FAISS index"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["ivf", "hnsw"],
        default="ivf",
        help="Target index type (default: ivf)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of original index"
    )
    
    args = parser.parse_args()
    
    optimize_index(
        index_path=args.index_path,
        target_type=args.type,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()

