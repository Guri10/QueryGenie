"""
Document Preprocessing Pipeline
Handles chunking, embedding, and FAISS index creation
"""

import json
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import re
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata"""
    chunk_id: str
    text: str
    paper_id: str
    paper_title: str
    authors: List[str]
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any]


class DocumentProcessor:
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            embedding_model: Hugging Face model for embeddings
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        print(f"Model loaded successfully!")
        
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk (defaults to self.chunk_size)
            overlap: Overlap between chunks (defaults to self.chunk_overlap)
            
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.chunk_overlap
            
        # Simple word-based chunking (can be improved with tokenization)
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            
            if end == len(words):
                break
                
            start = end - overlap
            
        return chunks
    
    def process_papers(self, papers_file: str) -> List[DocumentChunk]:
        """
        Process papers from JSON file into chunks
        
        Args:
            papers_file: Path to papers JSON file
            
        Returns:
            List of document chunks
        """
        print(f"Processing papers from: {papers_file}")
        
        with open(papers_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        all_chunks = []
        
        for paper_idx, paper in enumerate(tqdm(papers, desc="Processing papers")):
            # Combine title and abstract for chunking
            full_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
            
            # Clean and preprocess text
            full_text = self.clean_text(full_text)
            
            # Chunk the text
            text_chunks = self.chunk_text(full_text)
            
            # Create DocumentChunk objects
            for chunk_idx, chunk_text in enumerate(text_chunks):
                chunk = DocumentChunk(
                    chunk_id=f"{paper['id']}_chunk_{chunk_idx}",
                    text=chunk_text,
                    paper_id=paper['id'],
                    paper_title=paper['title'],
                    authors=paper['authors'],
                    chunk_index=chunk_idx,
                    total_chunks=len(text_chunks),
                    metadata={
                        'published': paper['published'],
                        'updated': paper['updated'],
                        'categories': paper['categories'],
                        'primary_category': paper['primary_category'],
                        'pdf_url': paper['pdf_url'],
                        'doi': paper['doi']
                    }
                )
                all_chunks.append(chunk)
        
        print(f"Created {len(all_chunks)} chunks from {len(papers)} papers")
        return all_chunks
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()
    
    def create_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[np.ndarray, List[DocumentChunk]]:
        """
        Create embeddings for document chunks
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Tuple of (embeddings_array, chunks_list)
        """
        print(f"Creating embeddings for {len(chunks)} chunks...")
        
        texts = [chunk.text for chunk in chunks]
        
        # Create embeddings in batches for memory efficiency
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedder.encode(batch_texts, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        print(f"Created embeddings with shape: {embeddings.shape}")
        
        return embeddings, chunks
    
    def build_faiss_index(self, 
                         embeddings: np.ndarray, 
                         chunks: List[DocumentChunk],
                         index_path: str = "data/faiss_index",
                         index_type: str = "auto") -> str:
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Embedding vectors
            chunks: Document chunks
            index_path: Path to save the index
            index_type: Type of index ('auto', 'flat', 'ivf', 'hnsw')
            
        Returns:
            Path to saved index
        """
        print("Building FAISS index...")
        
        # Import optimizer
        try:
            from faiss_optimizer import FAISSOptimizer
            use_optimizer = True
        except ImportError:
            print("Warning: faiss_optimizer not available, using flat index")
            use_optimizer = False
        
        dimension = embeddings.shape[1]
        
        # Create FAISS index
        if use_optimizer and index_type != "flat":
            print(f"Creating optimized {index_type} index...")
            index = FAISSOptimizer.create_optimized_index(
                embeddings,
                index_type=index_type
            )
        else:
            print("Creating flat index (exact search)...")
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, f"{index_path}.faiss")
        
        # Save chunks metadata
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                'chunk_id': chunk.chunk_id,
                'text': chunk.text,
                'paper_id': chunk.paper_id,
                'paper_title': chunk.paper_title,
                'authors': chunk.authors,
                'chunk_index': chunk.chunk_index,
                'total_chunks': chunk.total_chunks,
                'metadata': chunk.metadata
            })
        
        with open(f"{index_path}_chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"FAISS index saved to: {index_path}.faiss")
        print(f"Chunks metadata saved to: {index_path}_chunks.json")
        
        return index_path
    
    def process_and_index(self, papers_file: str, index_path: str = "data/faiss_index", index_type: str = "auto") -> str:
        """
        Complete pipeline: process papers and create FAISS index
        
        Args:
            papers_file: Path to papers JSON file
            index_path: Path to save the index
            index_type: Type of FAISS index ('auto', 'flat', 'ivf', 'hnsw')
            
        Returns:
            Path to saved index
        """
        print("Starting document processing pipeline...")
        
        # Process papers into chunks
        chunks = self.process_papers(papers_file)
        
        # Create embeddings
        embeddings, chunks = self.create_embeddings(chunks)
        
        # Build FAISS index
        index_path = self.build_faiss_index(embeddings, chunks, index_path, index_type)
        
        print("Pipeline completed successfully!")
        return index_path


def main():
    """Test the preprocessing pipeline"""
    import glob
    
    # Find the most recent papers file
    data_dir = "data"
    paper_files = glob.glob(f"{data_dir}/arxiv_papers_*.json")
    
    if not paper_files:
        print("No paper files found. Please run arxiv_downloader.py first.")
        return
    
    # Use the most recent file
    latest_file = max(paper_files, key=os.path.getctime)
    print(f"Using papers file: {latest_file}")
    
    # Process papers
    processor = DocumentProcessor()
    index_path = processor.process_and_index(latest_file)
    
    print(f"Index created at: {index_path}")


if __name__ == "__main__":
    main()
