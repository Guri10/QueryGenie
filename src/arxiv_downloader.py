"""
arXiv Paper Downloader
Downloads recent papers from arXiv for the RAG corpus
"""

import arxiv
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time
from tqdm import tqdm


class ArxivDownloader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_papers(self, 
                       categories: List[str] = ["cs.AI", "cs.CL", "cs.LG", "cs.IR"],
                       max_results: int = 100,
                       days_back: int = 30,
                       check_existing: bool = True) -> str:
        """
        Download recent papers from arXiv
        
        Args:
            categories: List of arXiv categories to search
            max_results: Maximum number of papers to download
            days_back: How many days back to search
            check_existing: Whether to check for existing papers to avoid duplicates
            
        Returns:
            Path to the downloaded papers JSON file
        """
        print(f"Downloading papers from arXiv...")
        print(f"Categories: {categories}")
        print(f"Max results: {max_results}")
        print(f"Days back: {days_back}")
        
        # Load existing papers to avoid duplicates
        existing_paper_ids = set()
        if check_existing:
            existing_paper_ids = self._load_existing_paper_ids()
            print(f"Found {len(existing_paper_ids)} existing papers to avoid duplicates")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_papers = []
        
        for category in categories:
            print(f"\nSearching category: {category}")
            
            # Create search query
            query = f"cat:{category} AND submittedDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
            
            try:
                # Search for papers
                search = arxiv.Search(
                    query=query,
                    max_results=max_results // len(categories),
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                papers_found = 0
                duplicates_skipped = 0
                for paper in tqdm(search.results(), desc=f"Downloading {category}"):
                    try:
                        # Check for duplicates
                        if paper.entry_id in existing_paper_ids:
                            duplicates_skipped += 1
                            continue
                            
                        paper_data = {
                            "id": paper.entry_id,
                            "title": paper.title,
                            "authors": [str(author) for author in paper.authors],
                            "abstract": paper.summary,
                            "published": paper.published.isoformat(),
                            "updated": paper.updated.isoformat(),
                            "categories": paper.categories,
                            "pdf_url": paper.pdf_url,
                            "doi": paper.doi,
                            "journal_ref": paper.journal_ref,
                            "primary_category": paper.primary_category
                        }
                        all_papers.append(paper_data)
                        existing_paper_ids.add(paper.entry_id)  # Track this paper
                        papers_found += 1
                        
                        # Small delay to be respectful to arXiv
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"Error processing paper {paper.entry_id}: {e}")
                        continue
                        
                print(f"Downloaded {papers_found} papers from {category}")
                if duplicates_skipped > 0:
                    print(f"Skipped {duplicates_skipped} duplicate papers from {category}")
                
            except Exception as e:
                print(f"Error searching category {category}: {e}")
                continue
        
        # Save papers to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arxiv_papers_{timestamp}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_papers, f, indent=2, ensure_ascii=False)
        
        print(f"\nDownloaded {len(all_papers)} papers total")
        print(f"Saved to: {filepath}")
        
        return filepath
    
    def _load_existing_paper_ids(self) -> set:
        """
        Load existing paper IDs from all previous downloads to avoid duplicates
        
        Returns:
            Set of existing paper IDs
        """
        existing_ids = set()
        
        # Find all existing paper files
        paper_files = [f for f in os.listdir(self.data_dir) 
                      if f.startswith('arxiv_papers_') and f.endswith('.json')]
        
        for filename in paper_files:
            filepath = os.path.join(self.data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
                    for paper in papers:
                        existing_ids.add(paper['id'])
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                continue
        
        return existing_ids
    
    def get_latest_papers_file(self) -> str:
        """Get the most recent papers file"""
        if not os.path.exists(self.data_dir):
            return None
            
        files = [f for f in os.listdir(self.data_dir) if f.startswith("arxiv_papers_") and f.endswith(".json")]
        if not files:
            return None
            
        # Sort by filename (which includes timestamp)
        files.sort(reverse=True)
        return os.path.join(self.data_dir, files[0])


def main():
    """Download papers for testing"""
    downloader = ArxivDownloader()
    
    # Download papers from recent AI/ML categories
    categories = [
        "cs.AI",      # Artificial Intelligence
        "cs.CL",      # Computation and Language
        "cs.LG",      # Machine Learning
        "cs.IR",      # Information Retrieval
        "cs.CV",      # Computer Vision
        "cs.NE"       # Neural and Evolutionary Computing
    ]
    
    papers_file = downloader.download_papers(
        categories=categories,
        max_results=200,  # 200 papers total
        days_back=60      # Last 60 days
    )
    
    print(f"Papers downloaded to: {papers_file}")


if __name__ == "__main__":
    main()
