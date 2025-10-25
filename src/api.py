"""
FastAPI Service for QueryGenie RAG Chatbot
Provides REST API endpoints for querying the RAG system
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import os
import json
from datetime import datetime

from faiss_manager import FAISSManager
from rag_pipeline import RAGPipeline
from arxiv_downloader import ArxivDownloader
from preprocessing import DocumentProcessor


# Pydantic models for API
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
    index_stats: Dict[str, Any]
    uptime: float


class MetricsResponse(BaseModel):
    total_queries: int
    average_retrieval_time: float
    average_generation_time: float
    average_total_time: float
    pipeline_stats: Dict[str, Any]


# Global variables
app = FastAPI(
    title="QueryGenie RAG Chatbot",
    description="A free, local-only RAG system using arXiv papers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="web"), name="static")

# Global RAG pipeline
rag_pipeline = None
start_time = time.time()
query_metrics = {
    "total_queries": 0,
    "retrieval_times": [],
    "generation_times": [],
    "total_times": []
}


def initialize_pipeline(use_llm: bool = False, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Initialize the RAG pipeline
    
    Args:
        use_llm: Whether to enable LLM generation
        model_name: Name of the LLM model to use
    """
    global rag_pipeline
    
    try:
        print("Initializing QueryGenie RAG Pipeline...")
        
        # Initialize FAISS manager
        faiss_manager = FAISSManager()
        
        if not faiss_manager.is_ready():
            print("FAISS index not found. Please run preprocessing first.")
            print("API will start in limited mode - some endpoints may not work.")
            return False
        
        # Initialize RAG pipeline with optional LLM
        rag_pipeline = RAGPipeline(
            faiss_manager,
            use_llm=use_llm,
            model_name=model_name
        )
        
        print("RAG Pipeline initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("API will start in limited mode - some endpoints may not work.")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    # Check environment variable for LLM usage
    use_llm = os.getenv("USE_LLM", "false").lower() == "true"
    model_name = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    if use_llm:
        print(f"🤖 LLM generation enabled with model: {model_name}")
    else:
        print("📚 Running in retrieval-only mode (set USE_LLM=true to enable LLM generation)")
    
    success = initialize_pipeline(use_llm=use_llm, model_name=model_name)
    if not success:
        print("Warning: RAG pipeline not initialized. Some endpoints may not work.")


@app.get("/")
async def root():
    """Serve the web interface"""
    return FileResponse("web/index.html")

@app.get("/api", response_model=Dict[str, str])
async def api_info():
    """API info endpoint"""
    return {
        "message": "QueryGenie RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "pipeline_status": "ready" if rag_pipeline is not None else "not_initialized"
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint that doesn't require RAG pipeline"""
    return {
        "message": "QueryGenie API is running",
        "timestamp": datetime.now().isoformat(),
        "status": "ok"
    }


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask a question to the RAG chatbot
    
    Args:
        request: Query request with question and parameters
        
    Returns:
        QueryResponse with answer and sources
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG pipeline not initialized. Please check if the index exists."
        )
    
    try:
        # Process query
        response = rag_pipeline.query(
            question=request.question,
            k=request.k,
            max_context_length=request.max_context_length,
            max_answer_length=request.max_answer_length
        )
        
        # Update metrics
        query_metrics["total_queries"] += 1
        query_metrics["retrieval_times"].append(response.retrieval_time)
        query_metrics["generation_times"].append(response.generation_time)
        query_metrics["total_times"].append(response.total_time)
        
        # Keep only last 100 queries for metrics
        for key in ["retrieval_times", "generation_times", "total_times"]:
            if len(query_metrics[key]) > 100:
                query_metrics[key] = query_metrics[key][-100:]
        
        return QueryResponse(
            answer=response.answer,
            sources=response.sources,
            retrieval_time=response.retrieval_time,
            generation_time=response.generation_time,
            total_time=response.total_time,
            model_used=response.model_used,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthResponse with system status
    """
    pipeline_ready = rag_pipeline is not None and rag_pipeline.faiss_manager.is_ready()
    
    index_stats = {}
    if rag_pipeline and rag_pipeline.faiss_manager.is_ready():
        index_stats = rag_pipeline.faiss_manager.get_stats()
    
    return HealthResponse(
        status="healthy" if pipeline_ready else "degraded",
        pipeline_ready=pipeline_ready,
        index_stats=index_stats,
        uptime=time.time() - start_time
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get system metrics
    
    Returns:
        MetricsResponse with performance metrics
    """
    total_queries = query_metrics["total_queries"]
    
    if total_queries == 0:
        return MetricsResponse(
            total_queries=0,
            average_retrieval_time=0.0,
            average_generation_time=0.0,
            average_total_time=0.0,
            pipeline_stats={}
        )
    
    avg_retrieval = sum(query_metrics["retrieval_times"]) / len(query_metrics["retrieval_times"])
    avg_generation = sum(query_metrics["generation_times"]) / len(query_metrics["generation_times"])
    avg_total = sum(query_metrics["total_times"]) / len(query_metrics["total_times"])
    
    pipeline_stats = {}
    if rag_pipeline:
        pipeline_stats = rag_pipeline.get_stats()
    
    return MetricsResponse(
        total_queries=total_queries,
        average_retrieval_time=avg_retrieval,
        average_generation_time=avg_generation,
        average_total_time=avg_total,
        pipeline_stats=pipeline_stats
    )


@app.post("/refresh")
async def refresh_index(background_tasks: BackgroundTasks):
    """
    Refresh the FAISS index with new papers
    
    Returns:
        Status message
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    # Add background task to refresh index
    background_tasks.add_task(refresh_index_background)
    
    return {"message": "Index refresh started in background"}


async def refresh_index_background():
    """Background task to refresh the index"""
    try:
        print("Starting background index refresh...")
        
        # Download new papers
        downloader = ArxivDownloader()
        papers_file = downloader.download_papers(
            categories=["cs.AI", "cs.CL", "cs.LG", "cs.IR", "cs.CV", "cs.NE"],
            max_results=200,
            days_back=30
        )
        
        # Process papers
        processor = DocumentProcessor()
        index_path = processor.process_and_index(papers_file)
        
        # Reload FAISS manager
        global rag_pipeline
        faiss_manager = FAISSManager()
        rag_pipeline = RAGPipeline(faiss_manager)
        
        print("Index refresh completed successfully!")
        
    except Exception as e:
        print(f"Error in background refresh: {e}")


@app.get("/stats")
async def get_stats():
    """
    Get detailed system statistics
    
    Returns:
        Detailed system statistics
    """
    if rag_pipeline is None:
        return {"error": "RAG pipeline not initialized"}
    
    return rag_pipeline.get_stats()


if __name__ == "__main__":
    import uvicorn
    
    # Check environment variable for LLM usage
    use_llm = os.getenv("USE_LLM", "false").lower() == "true"
    model_name = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    if use_llm:
        print(f"🤖 LLM generation enabled with model: {model_name}")
    else:
        print("📚 Running in retrieval-only mode")
        print("   To enable LLM generation: USE_LLM=true python src/api.py")
    
    # Initialize pipeline before starting server
    initialize_pipeline(use_llm=use_llm, model_name=model_name)
    
    # Start server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
