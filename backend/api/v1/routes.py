"""
API v1 Routes for QueryGenie
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Change to the project root directory for data access
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from faiss_manager import FAISSManager
from rag_pipeline import RAGPipeline
from arxiv_downloader import ArxivDownloader
from preprocessing import DocumentProcessor

router = APIRouter()

# Global variables for pipeline
rag_pipeline = None
faiss_manager = None

def initialize_pipeline():
    """Initialize the RAG pipeline"""
    global rag_pipeline, faiss_manager
    
    try:
        print("Initializing QueryGenie RAG Pipeline...")
        faiss_manager = FAISSManager()
        
        if not faiss_manager.is_ready():
            raise RuntimeError("FAISS index not found. Please run preprocessing first.")
        
        use_llm = os.getenv("USE_LLM", "false").lower() == "true"
        model_name = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        rag_pipeline = RAGPipeline(
            faiss_manager,
            use_llm=use_llm,
            model_name=model_name
        )
        
        print("RAG Pipeline initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return False

# Initialize pipeline on startup
initialize_pipeline()

# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the RAG system")
    k: int = Field(5, ge=1, le=20, description="Number of sources to retrieve")
    max_context_length: int = Field(5000, ge=100, le=10000, description="Maximum context length")
    max_answer_length: int = Field(200, ge=50, le=500, description="Maximum answer length")

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float
    total_time: float
    model_used: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    timestamp: str

class StatsResponse(BaseModel):
    status: str
    pipeline_stats: Dict[str, Any]
    timestamp: str

# Routes
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    pipeline_ready = rag_pipeline is not None and faiss_manager is not None and faiss_manager.is_ready()
    
    return HealthResponse(
        status="healthy" if pipeline_ready else "degraded",
        pipeline_ready=pipeline_ready,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
    )

@router.get("/metrics", response_model=StatsResponse)
async def get_metrics():
    """Get system metrics"""
    if not faiss_manager or not faiss_manager.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    try:
        stats = faiss_manager.get_stats()
        return StatsResponse(
            status="healthy",
            pipeline_stats=stats,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask a question to the RAG system"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        start_time = time.time()
        response = rag_pipeline.query(
            question=request.question,
            k=request.k,
            max_context_length=request.max_context_length,
            max_answer_length=request.max_answer_length
        )
        # Format sources for API response
        sources = []
        for i, source in enumerate(response.sources, 1):
            sources.append({
                "number": i,
                "paper_title": source["paper_title"],
                "authors": source["authors"],
                "paper_id": source["paper_id"],
                "similarity_score": source["similarity_score"],
                "text_preview": source["text_preview"]
            })
        
        return QueryResponse(
            answer=response.answer,
            sources=sources,
            retrieval_time=response.retrieval_time,
            generation_time=response.generation_time,
            total_time=response.total_time,
            model_used=response.model_used,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@router.post("/refresh")
async def refresh_index(background_tasks: BackgroundTasks):
    """Refresh the FAISS index"""
    if not faiss_manager:
        raise HTTPException(status_code=503, detail="FAISS manager not initialized")
    
    def refresh_task():
        try:
            # This would trigger a refresh of the index
            # Implementation depends on your refresh strategy
            pass
        except Exception as e:
            print(f"Error refreshing index: {e}")
    
    background_tasks.add_task(refresh_task)
    return {"message": "Index refresh started", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
