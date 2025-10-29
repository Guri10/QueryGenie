"""
Safe FAISS Index Builder with Batch Processing
Prevents hanging on large datasets
"""

import faiss
import numpy as np
import json
import os
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass
import time


@dataclass
class Chunk:
    chunk_id: str
    text: str
    paper_id: str
    paper_title: str
    authors: List[str]
    chunk_index: int
    total_chunks: int
    metadata: dict


def load_papers(papers_file: str):
    """Load papers from JSON file"""
    print(f"Loading papers from {papers_file}...")
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers")
    return papers


def chunk_papers(papers, chunk_size=1000, chunk_overlap=200):
    """Chunk papers using RecursiveCharacterTextSplitter"""
    print(f"\nChunking papers (size={chunk_size}, overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )
    
    chunks = []
    for paper in papers:
        # Combine title + abstract for chunking
        full_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        
        # Split into chunks
        text_chunks = text_splitter.split_text(full_text)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = Chunk(
                chunk_id=f"{paper['id']}_chunk_{i}",
                text=chunk_text,
                paper_id=paper['id'],
                paper_title=paper['title'],
                authors=paper.get('authors', []),
                chunk_index=i,
                total_chunks=len(text_chunks),
                metadata={'categories': paper.get('categories', [])}
            )
            chunks.append(chunk)
    
    print(f"Created {len(chunks)} chunks from {len(papers)} papers")
    return chunks


def create_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
    """Create embeddings in batches with progress tracking"""
    print(f"\nLoading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    
    texts = [chunk.text for chunk in chunks]
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"Creating embeddings for {len(texts)} chunks in {total_batches} batches...")
    
    all_embeddings = []
    start_time = time.time()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embedder.encode(batch_texts, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
        
        # Progress update every 10 batches
        batch_num = i // batch_size + 1
        if batch_num % 10 == 0 or batch_num == total_batches:
            elapsed = time.time() - start_time
            print(f"  Batch {batch_num}/{total_batches} ({i+len(batch_texts)}/{len(texts)} chunks) - {elapsed:.1f}s elapsed")
    
    embeddings = np.vstack(all_embeddings)
    print(f"✅ Embeddings created: shape {embeddings.shape}")
    return embeddings


def build_faiss_index_safe(embeddings, chunks, index_path="data/faiss_index", index_type="flat"):
    """
    Build FAISS index with safe batching to prevent hangs
    
    Args:
        embeddings: numpy array of embeddings
        chunks: list of Chunk objects
        index_path: path to save index
        index_type: 'flat', 'ivf', or 'hnsw'
    """
    print(f"\n{'='*60}")
    print(f"Building {index_type.upper()} FAISS index")
    print(f"{'='*60}")
    
    n_samples, dimension = embeddings.shape
    print(f"Vectors: {n_samples}, Dimension: {dimension}")
    
    # Normalize embeddings for cosine similarity
    print("Normalizing embeddings...")
    embeddings_normalized = embeddings.copy().astype('float32')
    faiss.normalize_L2(embeddings_normalized)
    
    # Create index based on type
    if index_type == "flat":
        print("Creating Flat index (exact search)...")
        index = faiss.IndexFlatIP(dimension)
        
    elif index_type == "ivf":
        print("Creating IVF index (approximate search)...")
        nlist = min(100, n_samples // 10)  # Adaptive cluster count
        nprobe = 10
        
        print(f"  Parameters: nlist={nlist}, nprobe={nprobe}")
        
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train with sample if dataset is large
        if n_samples > 10000:
            train_size = min(10000, n_samples)
            print(f"  Training on {train_size} samples...")
            train_data = embeddings_normalized[:train_size]
            index.train(train_data)
        else:
            print(f"  Training on all {n_samples} samples...")
            index.train(embeddings_normalized)
        
        index.nprobe = nprobe
        print("  ✅ Training complete")
        
    elif index_type == "hnsw":
        print("Creating HNSW index (fast approximate search)...")
        M = 16  # Reduced from 32 to prevent hangs
        print(f"  Parameters: M={M}, efConstruction=40")
        
        index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 16
        
    else:
        raise ValueError(f"Unknown index type: {index_type}. Use 'flat', 'ivf', or 'hnsw'")
    
    # Add vectors in batches to prevent hanging
    batch_size = 1000
    total_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"\nAdding {n_samples} vectors in {total_batches} batches of {batch_size}...")
    start_time = time.time()
    
    for i in range(0, n_samples, batch_size):
        batch = embeddings_normalized[i:i+batch_size]
        index.add(batch)
        
        batch_num = i // batch_size + 1
        elapsed = time.time() - start_time
        print(f"  Batch {batch_num}/{total_batches} ({i+len(batch)}/{n_samples} vectors) - {elapsed:.1f}s")
    
    print(f"✅ All vectors added in {time.time() - start_time:.1f}s")
    print(f"Index size: {index.ntotal} vectors")
    
    # Save index
    print(f"\nSaving index to {index_path}.faiss...")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, f"{index_path}.faiss")
    print("✅ Index saved")
    
    # Save chunks metadata
    print(f"Saving chunks to {index_path}_chunks.json...")
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
    print("✅ Chunks saved")
    
    print(f"\n{'='*60}")
    print(f"✅ FAISS INDEX BUILD COMPLETE!")
    print(f"{'='*60}")
    print(f"Index file:  {index_path}.faiss")
    print(f"Chunks file: {index_path}_chunks.json")
    print(f"Total time:  {time.time() - start_time:.1f}s")
    
    return index_path


def main():
    """Main pipeline"""
    import sys
    
    # Configuration
    papers_file = "data/arxiv_papers_merged.json"
    index_path = "data/faiss_index"
    index_type = sys.argv[1] if len(sys.argv) > 1 else "flat"
    
    if index_type not in ['flat', 'ivf', 'hnsw']:
        print(f"❌ Invalid index type: {index_type}")
        print("Usage: python preprocessing_safe.py [flat|ivf|hnsw]")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"QueryGenie - Safe FAISS Index Builder")
    print(f"{'='*60}")
    print(f"Papers file: {papers_file}")
    print(f"Index type:  {index_type.upper()}")
    print(f"Index path:  {index_path}")
    print(f"{'='*60}\n")
    
    total_start = time.time()
    
    # Step 1: Load papers
    papers = load_papers(papers_file)
    
    # Step 2: Chunk papers
    chunks = chunk_papers(papers, chunk_size=1000, chunk_overlap=200)
    
    # Step 3: Create embeddings
    embeddings = create_embeddings(chunks, batch_size=32)
    
    # Step 4: Build FAISS index
    build_faiss_index_safe(embeddings, chunks, index_path, index_type)
    
    total_time = time.time() - total_start
    print(f"\n🎉 PIPELINE COMPLETE IN {total_time:.1f}s ({total_time/60:.1f} minutes)")


if __name__ == "__main__":
    main()

