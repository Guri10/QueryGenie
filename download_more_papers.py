#!/usr/bin/env python3
"""
Custom Paper Downloader
Download papers from specific categories with custom parameters
"""

from src.arxiv_downloader import ArxivDownloader
from src.database_cleaner import DatabaseCleaner
from src.preprocessing import DocumentProcessor
import argparse


def download_papers(categories, max_results, days_back, update_index=True):
    """
    Download papers and optionally update the index
    
    Args:
        categories: List of arXiv categories
        max_results: Maximum number of papers
        days_back: How many days back to search
        update_index: Whether to update FAISS index after download
    """
    print("🚀 Starting paper download...")
    
    # Step 1: Download papers
    downloader = ArxivDownloader()
    papers_file = downloader.download_papers(
        categories=categories,
        max_results=max_results,
        days_back=days_back,
        check_existing=True  # Enable duplicate prevention
    )
    
    print(f"✅ Papers downloaded to: {papers_file}")
    
    if update_index:
        print("\n🔄 Updating database...")
        
        # Step 2: Merge with existing database
        cleaner = DatabaseCleaner()
        merged_file = cleaner.merge_all_papers()
        
        if merged_file:
            print(f"Database merged to: {merged_file}")
            
            # Step 3: Update FAISS index
            print("\n🔄 Updating FAISS index...")
            processor = DocumentProcessor()
            index_path = processor.process_and_index(merged_file)
            
            print(f"Index updated at: {index_path}")
            print("\nDatabase expansion complete!")
        else:
            print("Failed to merge database")
    else:
        print("ℹ️ Skipping index update (use --update-index to enable)")


def main():
    parser = argparse.ArgumentParser(description="Download more papers for QueryGenie")
    
    # Category options
    parser.add_argument('--categories', nargs='+', 
                       default=['cs.SE', 'cs.DC', 'cs.CR', 'cs.DB', 'cs.HC', 'cs.RO'],
                       help='arXiv categories to download from')
    
    # Download parameters
    parser.add_argument('--max-results', type=int, default=200,
                       help='Maximum number of papers to download')
    parser.add_argument('--days-back', type=int, default=90,
                       help='How many days back to search')
    
    # Index update
    parser.add_argument('--update-index', action='store_true', default=True,
                       help='Update FAISS index after download')
    parser.add_argument('--no-update', action='store_true',
                       help='Skip index update')
    
    args = parser.parse_args()
    
    # Handle update index flag
    update_index = args.update_index and not args.no_update
    
    # Available categories
    available_categories = {
        'cs.AI': 'Artificial Intelligence',
        'cs.CL': 'Computation and Language', 
        'cs.LG': 'Machine Learning',
        'cs.IR': 'Information Retrieval',
        'cs.CV': 'Computer Vision',
        'cs.NE': 'Neural and Evolutionary Computing',
        'cs.SE': 'Software Engineering',
        'cs.DC': 'Distributed Computing',
        'cs.CR': 'Cryptography',
        'cs.DB': 'Databases',
        'cs.HC': 'Human-Computer Interaction',
        'cs.RO': 'Robotics',
        'cs.CG': 'Computer Graphics',
        'cs.DM': 'Discrete Mathematics',
        'cs.GT': 'Game Theory',
        'cs.LO': 'Logic',
        'cs.PL': 'Programming Languages',
        'cs.SY': 'Systems and Control'
    }
    
    print("📚 Available Categories:")
    for cat, desc in available_categories.items():
        print(f"  {cat}: {desc}")
    
    print(f"\n🎯 Downloading from: {args.categories}")
    print(f"📊 Max results: {args.max_results}")
    print(f"📅 Days back: {args.days_back}")
    print(f"🔄 Update index: {update_index}")
    
    download_papers(
        categories=args.categories,
        max_results=args.max_results,
        days_back=args.days_back,
        update_index=update_index
    )


if __name__ == "__main__":
    main()
