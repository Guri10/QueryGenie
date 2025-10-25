"""
Download Top Cited Papers from arXiv
Focuses on high-quality, influential papers across 20 years
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import arxiv
from tqdm import tqdm


class TopPapersDownloader:
    """
    Download top cited/influential papers from arXiv
    
    Strategy:
    1. Download most recent papers (proxy for citations - newer papers get cited more)
    2. Sort by submission date (recent = more likely to be cited)
    3. Focus on core CS categories
    4. Aim for quality over quantity
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_historical_papers(
        self,
        categories: List[str],
        total_papers: int = 10000,
        years_back: int = 20,
        additional_categories: List[str] = None
    ) -> str:
        """
        Download top papers from last N years
        
        Args:
            categories: Base categories to download from
            total_papers: Total number of papers to download
            years_back: How many years to go back
            additional_categories: Extra categories to add
            
        Returns:
            Path to saved papers file
        """
        # Combine categories
        all_categories = categories.copy()
        if additional_categories:
            all_categories.extend(additional_categories)
        
        print(f"\n{'='*60}")
        print(f"Downloading Top {total_papers} Papers from Last {years_back} Years")
        print(f"{'='*60}")
        print(f"Categories: {', '.join(all_categories)}")
        print(f"Target: {total_papers} papers")
        print(f"Strategy: Quality over quantity (top cited)")
        print(f"{'='*60}\n")
        
        # Calculate papers per year
        papers_per_year = total_papers // years_back
        print(f"Target: ~{papers_per_year} papers per year\n")
        
        # Load existing papers to avoid duplicates
        existing_paper_ids = self._load_existing_paper_ids()
        print(f"Found {len(existing_paper_ids)} existing papers to avoid duplicates\n")
        
        all_papers = []
        
        # Download year by year (most recent first)
        end_date = datetime.now()
        
        for year_offset in range(years_back):
            year_end = end_date - timedelta(days=365 * year_offset)
            year_start = year_end - timedelta(days=365)
            
            year = year_end.year
            print(f"\n{'─'*60}")
            print(f"Year {year}: {year_start.strftime('%Y-%m-%d')} to {year_end.strftime('%Y-%m-%d')}")
            print(f"{'─'*60}")
            
            year_papers = self._download_year(
                all_categories,
                year_start,
                year_end,
                papers_per_year,
                existing_paper_ids
            )
            
            all_papers.extend(year_papers)
            print(f"✅ Year {year}: Downloaded {len(year_papers)} papers (Total: {len(all_papers)})")
            
            # Stop if we've reached target
            if len(all_papers) >= total_papers:
                print(f"\n✅ Reached target of {total_papers} papers!")
                break
        
        # Save papers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arxiv_papers_historical_{timestamp}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_papers, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Download Complete!")
        print(f"{'='*60}")
        print(f"Total papers downloaded: {len(all_papers)}")
        print(f"Saved to: {filepath}")
        print(f"{'='*60}\n")
        
        return filepath
    
    def _download_year(
        self,
        categories: List[str],
        start_date: datetime,
        end_date: datetime,
        target_papers: int,
        existing_ids: set
    ) -> List[Dict[str, Any]]:
        """Download papers for a specific year"""
        
        papers_per_category = max(1, target_papers // len(categories))
        year_papers = []
        
        for category in categories:
            print(f"  📥 {category}: Downloading ~{papers_per_category} papers...")
            
            # Create search query
            query = f"cat:{category} AND submittedDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
            
            try:
                # Search for papers (sort by submitted date - recent first)
                search = arxiv.Search(
                    query=query,
                    max_results=papers_per_category * 2,  # Get extra to account for duplicates
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                papers_found = 0
                duplicates_skipped = 0
                
                for paper in search.results():
                    try:
                        # Check for duplicates
                        if paper.entry_id in existing_ids:
                            duplicates_skipped += 1
                            continue
                        
                        # Skip if we have enough for this category
                        if papers_found >= papers_per_category:
                            break
                        
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
                        
                        year_papers.append(paper_data)
                        existing_ids.add(paper.entry_id)
                        papers_found += 1
                        
                        # Small delay to be respectful to arXiv
                        time.sleep(0.3)
                        
                    except Exception as e:
                        print(f"    ⚠️  Error processing paper: {e}")
                        continue
                
                print(f"    ✅ {category}: {papers_found} papers downloaded", end="")
                if duplicates_skipped > 0:
                    print(f" ({duplicates_skipped} duplicates skipped)")
                else:
                    print()
                
            except Exception as e:
                print(f"    ❌ Error searching {category}: {e}")
                continue
        
        return year_papers
    
    def _load_existing_paper_ids(self) -> set:
        """Load existing paper IDs to avoid duplicates"""
        existing_ids = set()
        
        try:
            paper_files = [
                f for f in os.listdir(self.data_dir) 
                if f.startswith('arxiv_papers_') and f.endswith('.json')
            ]
            
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
        except Exception as e:
            print(f"Warning: Error loading existing papers: {e}")
        
        return existing_ids


def main():
    parser = argparse.ArgumentParser(
        description="Download top cited papers from arXiv (last 20 years)"
    )
    parser.add_argument(
        "--total",
        type=int,
        default=10000,
        help="Total number of papers to download (default: 10000)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=20,
        help="Number of years to go back (default: 20)"
    )
    parser.add_argument(
        "--categories",
        nargs='+',
        default=["cs.AI", "cs.CL", "cs.LG", "cs.IR", "cs.CV", "cs.NE", 
                 "cs.SE", "cs.DC", "cs.CR", "cs.DB", "cs.HC", "cs.RO"],
        help="arXiv categories to download from"
    )
    parser.add_argument(
        "--add-categories",
        nargs='+',
        default=None,
        help="Additional categories to add (e.g., cs.DS cs.PL cs.GT)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Automatically merge with existing papers after download"
    )
    parser.add_argument(
        "--update-index",
        action="store_true",
        help="Automatically update FAISS index after download and merge"
    )
    
    args = parser.parse_args()
    
    # Download papers
    downloader = TopPapersDownloader()
    
    papers_file = downloader.download_historical_papers(
        categories=args.categories,
        total_papers=args.total,
        years_back=args.years,
        additional_categories=args.add_categories
    )
    
    # Merge if requested
    if args.merge:
        print("\n" + "="*60)
        print("Merging with existing database...")
        print("="*60)
        
        from database_cleaner import DatabaseCleaner
        cleaner = DatabaseCleaner()
        merged_file = cleaner.merge_all_papers()
        
        if merged_file:
            print(f"✅ Database merged to: {merged_file}")
            
            # Update index if requested
            if args.update_index:
                print("\n" + "="*60)
                print("Updating FAISS index...")
                print("="*60)
                
                from preprocessing import DocumentProcessor
                processor = DocumentProcessor()
                
                # Use optimized index for large dataset
                index_type = "ivf" if args.total < 50000 else "hnsw"
                print(f"Using {index_type.upper()} index for {args.total} papers")
                
                index_path = processor.process_and_index(
                    merged_file,
                    index_type=index_type
                )
                
                print(f"✅ Index updated at: {index_path}")
                print("\n" + "="*60)
                print("Database expansion complete!")
                print("="*60)
        else:
            print("❌ Failed to merge database")
    else:
        print("\nℹ️  To merge and update index, run:")
        print(f"  python {__file__} --total {args.total} --years {args.years} --merge --update-index")


if __name__ == "__main__":
    main()

