#!/usr/bin/env python3
"""
Nightly Index Refresh Script
Downloads new papers and rebuilds the FAISS index
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from arxiv_downloader import ArxivDownloader
from preprocessing import DocumentProcessor
from faiss_manager import FAISSManager


def setup_logging():
    """Setup logging for the refresh script"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "refresh.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def refresh_index():
    """Main function to refresh the index"""
    logger = setup_logging()
    
    try:
        logger.info("Starting nightly index refresh...")
        
        # Step 1: Download new papers
        logger.info("Downloading new papers from arXiv...")
        downloader = ArxivDownloader()
        
        papers_file = downloader.download_papers(
            categories=[
                "cs.AI",      # Artificial Intelligence
                "cs.CL",      # Computation and Language
                "cs.LG",      # Machine Learning
                "cs.IR",      # Information Retrieval
                "cs.CV",      # Computer Vision
                "cs.NE",      # Neural and Evolutionary Computing
                "cs.RO",      # Robotics
                "cs.CY"       # Cryptography and Security
            ],
            max_results=300,  # More papers for better coverage
            days_back=7       # Last week's papers
        )
        
        logger.info(f"Downloaded papers saved to: {papers_file}")
        
        # Step 2: Process papers and create embeddings
        logger.info("Processing papers and creating embeddings...")
        processor = DocumentProcessor(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=512,
            chunk_overlap=50
        )
        
        index_path = processor.process_and_index(papers_file)
        logger.info(f"New index created at: {index_path}")
        
        # Step 3: Verify the new index
        logger.info("Verifying new index...")
        faiss_manager = FAISSManager(index_path)
        
        if faiss_manager.is_ready():
            stats = faiss_manager.get_stats()
            logger.info(f"Index verification successful!")
            logger.info(f"Total chunks: {stats['total_chunks']}")
            logger.info(f"Total papers: {stats['total_papers']}")
            logger.info(f"Index vectors: {stats['index_vectors']}")
        else:
            logger.error("Index verification failed!")
            return False
        
        # Step 4: Clean up old files (keep last 3 versions)
        logger.info("Cleaning up old files...")
        cleanup_old_files()
        
        logger.info("Nightly index refresh completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during index refresh: {e}")
        return False


def cleanup_old_files():
    """Clean up old papers and index files, keeping only the last 3 versions"""
    data_dir = Path("data")
    
    if not data_dir.exists():
        return
    
    # Get all papers files
    papers_files = list(data_dir.glob("arxiv_papers_*.json"))
    papers_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Keep only the last 3 papers files
    for old_file in papers_files[3:]:
        try:
            old_file.unlink()
            print(f"Deleted old papers file: {old_file}")
        except Exception as e:
            print(f"Error deleting {old_file}: {e}")
    
    # Get all index files
    index_files = list(data_dir.glob("faiss_index_*.faiss"))
    index_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Keep only the last 3 index files
    for old_file in index_files[3:]:
        try:
            old_file.unlink()
            # Also delete corresponding chunks file
            chunks_file = old_file.with_suffix("_chunks.json")
            if chunks_file.exists():
                chunks_file.unlink()
            print(f"Deleted old index file: {old_file}")
        except Exception as e:
            print(f"Error deleting {old_file}: {e}")


def test_index():
    """Test the current index with a sample query"""
    logger = setup_logging()
    
    try:
        logger.info("Testing current index...")
        
        faiss_manager = FAISSManager()
        
        if not faiss_manager.is_ready():
            logger.error("No index found to test")
            return False
        
        # Test search
        test_query = "machine learning neural networks"
        results = faiss_manager.search(test_query, k=3)
        
        if results:
            logger.info(f"Index test successful! Found {len(results)} results for test query")
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: {result['paper_title'][:50]}...")
            return True
        else:
            logger.error("Index test failed - no results returned")
            return False
            
    except Exception as e:
        logger.error(f"Error testing index: {e}")
        return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QueryGenie Index Refresh Script")
    parser.add_argument("--test", action="store_true", help="Test current index")
    parser.add_argument("--force", action="store_true", help="Force refresh even if recent")
    
    args = parser.parse_args()
    
    if args.test:
        success = test_index()
        sys.exit(0 if success else 1)
    
    # Check if we should skip refresh (if done recently)
    if not args.force:
        last_refresh_file = Path("data/last_refresh.txt")
        if last_refresh_file.exists():
            try:
                with open(last_refresh_file, 'r') as f:
                    last_refresh = datetime.fromisoformat(f.read().strip())
                
                # Skip if refreshed within last 20 hours
                if (datetime.now() - last_refresh).total_seconds() < 20 * 3600:
                    print("Index was refreshed recently, skipping...")
                    sys.exit(0)
            except Exception:
                pass  # Continue with refresh if we can't read the file
    
    # Perform refresh
    success = refresh_index()
    
    if success:
        # Record successful refresh
        last_refresh_file = Path("data/last_refresh.txt")
        last_refresh_file.parent.mkdir(exist_ok=True)
        with open(last_refresh_file, 'w') as f:
            f.write(datetime.now().isoformat())
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
