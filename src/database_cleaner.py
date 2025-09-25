"""
Database Cleaner
Removes duplicates and merges paper databases
"""

import json
import os
from typing import List, Dict, Any, Set
from collections import defaultdict


class DatabaseCleaner:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        
    def clean_duplicates(self, papers_file: str) -> str:
        """
        Remove duplicates from a papers file
        
        Args:
            papers_file: Path to papers JSON file
            
        Returns:
            Path to cleaned papers file
        """
        print(f"Cleaning duplicates from: {papers_file}")
        
        with open(papers_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        print(f"Original papers: {len(papers)}")
        
        # Remove duplicates based on paper ID
        seen_ids = set()
        unique_papers = []
        
        for paper in papers:
            paper_id = paper['id']
            if paper_id not in seen_ids:
                seen_ids.add(paper_id)
                unique_papers.append(paper)
        
        print(f"Unique papers: {len(unique_papers)}")
        print(f"Duplicates removed: {len(papers) - len(unique_papers)}")
        
        # Save cleaned papers
        base_name = os.path.splitext(papers_file)[0]
        cleaned_file = f"{base_name}_cleaned.json"
        
        with open(cleaned_file, 'w', encoding='utf-8') as f:
            json.dump(unique_papers, f, indent=2, ensure_ascii=False)
        
        print(f"Cleaned papers saved to: {cleaned_file}")
        return cleaned_file
    
    def merge_all_papers(self, output_file: str = None) -> str:
        """
        Merge all paper files into one, removing duplicates
        
        Args:
            output_file: Output file path (optional)
            
        Returns:
            Path to merged papers file
        """
        print("Merging all paper files...")
        
        # Find all paper files
        paper_files = [f for f in os.listdir(self.data_dir) 
                      if f.startswith('arxiv_papers_') and f.endswith('.json')]
        
        if not paper_files:
            print("No paper files found!")
            return None
        
        print(f"Found {len(paper_files)} paper files")
        
        # Load all papers
        all_papers = []
        seen_ids = set()
        
        for filename in sorted(paper_files):
            filepath = os.path.join(self.data_dir, filename)
            print(f"Processing {filename}...")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
                
                for paper in papers:
                    paper_id = paper['id']
                    if paper_id not in seen_ids:
                        seen_ids.add(paper_id)
                        all_papers.append(paper)
                
                print(f"  Added {len(papers)} papers, {len(seen_ids)} unique total")
                
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                continue
        
        print(f"\nTotal unique papers: {len(all_papers)}")
        
        # Save merged papers
        if output_file is None:
            output_file = os.path.join(self.data_dir, "arxiv_papers_merged.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_papers, f, indent=2, ensure_ascii=False)
        
        print(f"Merged papers saved to: {output_file}")
        return output_file
    
    def analyze_duplicates(self) -> Dict[str, Any]:
        """
        Analyze duplicates across all paper files
        
        Returns:
            Dictionary with duplicate analysis
        """
        print("Analyzing duplicates...")
        
        # Find all paper files
        paper_files = [f for f in os.listdir(self.data_dir) 
                      if f.startswith('arxiv_papers_') and f.endswith('.json')]
        
        if not paper_files:
            return {"error": "No paper files found"}
        
        # Load all papers
        all_papers = []
        file_stats = {}
        
        for filename in sorted(paper_files):
            filepath = os.path.join(self.data_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
                
                all_papers.extend(papers)
                file_stats[filename] = len(papers)
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        # Analyze duplicates
        paper_ids = [paper['id'] for paper in all_papers]
        unique_ids = set(paper_ids)
        
        # Count duplicates
        id_counts = defaultdict(int)
        for paper_id in paper_ids:
            id_counts[paper_id] += 1
        
        duplicates = {pid: count for pid, count in id_counts.items() if count > 1}
        
        analysis = {
            "total_files": len(paper_files),
            "total_papers": len(all_papers),
            "unique_papers": len(unique_ids),
            "duplicates": len(duplicates),
            "duplicate_count": len(all_papers) - len(unique_ids),
            "file_stats": file_stats,
            "duplicate_ids": list(duplicates.keys())[:10]  # First 10 duplicate IDs
        }
        
        return analysis


def main():
    """Clean and merge database"""
    cleaner = DatabaseCleaner()
    
    # Analyze current state
    print("=== Database Analysis ===")
    analysis = cleaner.analyze_duplicates()
    
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print(f"Total files: {analysis['total_files']}")
    print(f"Total papers: {analysis['total_papers']}")
    print(f"Unique papers: {analysis['unique_papers']}")
    print(f"Duplicates: {analysis['duplicates']}")
    print(f"Duplicate count: {analysis['duplicate_count']}")
    
    print("\nFile statistics:")
    for filename, count in analysis['file_stats'].items():
        print(f"  {filename}: {count} papers")
    
    if analysis['duplicates'] > 0:
        print(f"\nSample duplicate IDs: {analysis['duplicate_ids']}")
        
        # Merge all papers
        print("\n=== Merging Papers ===")
        merged_file = cleaner.merge_all_papers()
        
        if merged_file:
            print(f"✅ Database cleaned and merged to: {merged_file}")
        else:
            print("❌ Failed to merge papers")
    else:
        print("✅ No duplicates found!")


if __name__ == "__main__":
    main()
