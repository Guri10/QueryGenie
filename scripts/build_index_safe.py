"""
Safe FAISS Index Builder
Builds index in smaller batches to avoid memory issues
"""

import sys
import os
import json
import numpy as np
import faiss
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import DocumentProcessor


def build_index_safe(
    papers_file: str = "data/arxiv_papers_merged.json",
    index_type: str = "flat",
    batch_size: int = 1000
):
    """
    Build FAISS index safely in batches
    
    Args:
        papers_file: Path to papers JSON file
        index_type: Type of index ('flat', 'ivf', 'hnsw')
        batch_size: Process papers in batches of this size
    """
    print("="*60)
    print("Safe FAISS Index Builder")
    print("="*60)
    print(f"Papers file: {papers_file}")
    print(f"Index type: {index_type}")
    print(f"Batch size: {batch_size}")
    print("="*60)
    print()
    
    # Load papers
    print("Loading papers...")
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"✅ Loaded {len(papers)} papers")
    print()
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Process all papers into chunks
    print("Processing papers into chunks...")
    all_chunks = []
    
    for paper in tqdm(papers, desc="Processing papers"):
        full_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        full_text = processor.clean_text(full_text)
        text_chunks = processor.chunk_text(full_text)
        
        for chunk_idx, chunk_text in enumerate(text_chunks):
            from preprocessing import DocumentChunk
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
    
    print(f"✅ Created {len(all_chunks)} chunks")
    print()
    
    # Create embeddings in batches
    print("Creating embeddings in batches...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding batches"):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_texts = [chunk.text for chunk in batch_chunks]
        
        # Generate embeddings for this batch
        batch_embeddings = processor.embedder.encode(
            batch_texts,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False
        )
        
        all_embeddings.append(batch_embeddings)
    
    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)
    print(f"✅ Created embeddings with shape: {embeddings.shape}")
    print()
    
    # Build FAISS index
    print(f"Building {index_type.upper()} FAISS index...")
    dimension = embeddings.shape[1]
    
    if index_type == "flat":
        # Simple flat index
        index = faiss.IndexFlatIP(dimension)
        print("Using Flat index (exact search)")
        
    elif index_type == "ivf":
        # IVF index for better performance
        print("Using IVF index (approximate search, 2-5x faster)")
        nlist = min(100, int(np.sqrt(len(all_chunks))))
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train the index
        print(f"Training IVF index with {nlist} clusters...")
        normalized_train = embeddings.copy().astype('float32')
        faiss.normalize_L2(normalized_train)
        index.train(normalized_train)
        index.nprobe = max(10, nlist // 10)
        print(f"✅ Training complete (nprobe={index.nprobe})")
        
    elif index_type == "hnsw":
        # HNSW index for maximum speed
        print("Using HNSW index (very fast approximate search)")
        M = 32
        index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 16
        
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Normalize and add vectors
    print("Adding vectors to index...")
    normalized = embeddings.copy().astype('float32')
    faiss.normalize_L2(normalized)
    
    # Add in batches to avoid memory issues
    for i in tqdm(range(0, len(normalized), batch_size), desc="Adding to index"):
        batch = normalized[i:i + batch_size]
        index.add(batch)
    
    print(f"✅ Added {index.ntotal} vectors to index")
    print()
    
    # Save index
    index_path = "data/faiss_index"
    print(f"Saving index to {index_path}.faiss...")
    faiss.write_index(index, f"{index_path}.faiss")
    print("✅ Index saved")
    
    # Save metadata
    print(f"Saving metadata to {index_path}_chunks.json...")
    chunks_data = []
    for chunk in tqdm(all_chunks, desc="Saving metadata"):
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
    
    print("✅ Metadata saved")
    print()
    
    print("="*60)
    print("✅ INDEX BUILD COMPLETE!")
    print("="*60)
    print(f"Total papers: {len(papers)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Index vectors: {index.ntotal}")
    print(f"Index type: {index_type.upper()}")
    print(f"Files created:")
    print(f"  - {index_path}.faiss")
    print(f"  - {index_path}_chunks.json")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index safely")
    parser.add_argument("--papers", default="data/arxiv_papers_merged.json", help="Papers file")
    parser.add_argument("--type", choices=["flat", "ivf", "hnsw"], default="ivf", help="Index type")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    
    args = parser.parse_args()
    
    build_index_safe(
        papers_file=args.papers,
        index_type=args.type,
        batch_size=args.batch_size
    )

