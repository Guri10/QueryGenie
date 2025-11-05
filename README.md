# QueryGenie RAG Chatbot

A **completely free, local-only** Retrieval-Augmented Generation (RAG) chatbot that uses arXiv papers as its knowledge base. Built with open-source components and designed to run entirely on your local machine with zero costs.

## 🚀 Features

- **Zero Cost**: Uses only free, open-source models and libraries
- **Local Only**: Runs entirely on your machine, no external APIs
- **arXiv Integration**: Automatically downloads and indexes recent research papers
- **Fast Retrieval**: FAISS-based similarity search for efficient document retrieval
- **Smart Retrieval**: Intelligent document retrieval with similarity scoring
- **Optional LLM Generation**: AI-powered answer synthesis with local models (TinyLlama, Phi-3, Mistral)
- **REST API**: FastAPI-based service with comprehensive endpoints
- **Docker Support**: Easy deployment with Docker and docker-compose
- **Auto-Refresh**: Nightly cron job to update the knowledge base
- **Evaluation Tools**: Built-in metrics for hit@k, latency, and retrieval quality

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   arXiv Papers  │───▶│  Preprocessing  │───▶│  FAISS Index    │
│   (Downloader)  │    │  (Chunking +    │    │  (Embeddings)   │
│                 │    │   Embeddings)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐           │
│   User Query    │───▶│   RAG Pipeline  │◀──────────┘
│                 │    │  (Retrieval +   │
│                 │    │   Generation)   │
└─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   FastAPI       │
                       │   (REST API)    │
                       └─────────────────┘
```

## 📋 Requirements

- **Python 3.9+**
- **8-16 GB RAM** (recommended for smooth operation)
- **macOS/Linux/Windows** (tested on macOS M2)
- **Docker** (optional, for containerized deployment)

## 🛠️ Installation

### Option 1: Direct Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/QueryGenie.git
   cd QueryGenie
   ```

2. **Create virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download and process papers**

   ```bash
   python src/arxiv_downloader.py
   python src/preprocessing.py
   ```

5. **Start the API server**

   ```bash
   # Using the new backend structure
   python backend/main.py

   # Or using the legacy API (alternative)
   python src/api.py
   ```

### Option 2: Docker Deployment

1. **Build and run with Docker Compose**

   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **Initialize the system** (first time only)
   ```bash
   # Download papers and create index
   docker-compose -f docker-compose.prod.yml exec backend python src/arxiv_downloader.py
   docker-compose -f docker-compose.prod.yml exec backend python src/preprocessing.py
   ```

## 🚀 Quick Start

### 1. Download Papers

```bash
python src/arxiv_downloader.py
```

This downloads recent papers from arXiv (AI, ML, NLP, CV categories).

### 2. Create Index

```bash
python src/preprocessing.py
```

This processes papers, creates embeddings, and builds the FAISS index.

### 3. Start the API

```bash
# Using the new backend structure (recommended)
python backend/main.py

# Or using the legacy API
python src/api.py
```

The API will be available at `http://localhost:8000`

- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/v1/health`

### 4. Query the System

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the latest advances in transformer architectures?"}'
```

## 📚 API Endpoints

### Core Endpoints

- **`POST /api/v1/ask`** - Ask a question to the RAG system
- **`GET /api/v1/health`** - Health check and system status
- **`GET /api/v1/metrics`** - Performance metrics and statistics
- **`POST /api/v1/refresh`** - Trigger index refresh

**Note:** The API uses versioned endpoints under `/api/v1/`. For interactive API documentation, visit `http://localhost:8000/docs`.

### Example Usage

#### Ask a Question

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "How do attention mechanisms work in transformers?",
       "k": 5,
       "max_context_length": 5000,
       "max_answer_length": 300
     }'
```

#### Check System Health

```bash
curl http://localhost:8000/api/v1/health
```

#### Get Performance Metrics

```bash
curl http://localhost:8000/api/v1/metrics
```

## 🔧 Configuration

### Model Configuration

The system uses these free, open-source models:

- **Embeddings**: `sentence-transformers/all-MiniLLM-L6-v2` (384 dimensions, ~90MB)
- **Retrieval**: FAISS-based similarity search with sentence-transformers
- **Generation** (Optional): Local LLM via llama.cpp (TinyLlama, Phi-3, or Mistral)

### Running Modes

#### Retrieval-Only Mode (Default)

Fast and lightweight - returns formatted context from retrieved papers.

```bash
# Using new backend structure (recommended)
python backend/main.py

# Or using legacy API
python src/api.py
```

#### LLM Generation Mode (Optional)

AI-powered answer synthesis using local models.

```bash
# Install LLM dependencies first
pip install llama-cpp-python huggingface-hub

# Enable LLM generation (new backend)
USE_LLM=true python backend/main.py

# Use specific model
USE_LLM=true LLM_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0" python backend/main.py

# Or using legacy API
USE_LLM=true LLM_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0" python src/api.py
```

**Supported Models:**

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Fastest, ~600MB)
- `microsoft/phi-2` (Better quality, ~2.3GB)
- `mistralai/Mistral-7B-Instruct-v0.2` (Best quality, ~4GB)

📖 **See [SETUP_LLM.md](SETUP_LLM.md) for detailed LLM setup instructions.**

### Customization

You can modify the models in the source code:

```python
# In src/preprocessing.py
processor = DocumentProcessor(
    embedding_model="sentence-transformers/all-MiniLLM-L6-v2",  # Change this
    chunk_size=512,
    chunk_overlap=50
)

# In src/rag_pipeline.py
rag_pipeline = RAGPipeline(
    faiss_manager,
    use_llm=True,  # Enable LLM generation
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change this
)
```

## 📊 Evaluation

### Run Evaluation

```bash
python src/evaluation.py
```

### Custom Test Queries

Edit `test_queries.json` to add your own test questions.

### Evaluation Metrics

- **Latency**: Average response time (retrieval + generation)
- **Hit@k**: Percentage of queries with relevant results in top-k
- **Retrieval Quality**: Average similarity scores and diversity

## 🔄 Automated Updates

### Setup Nightly Refresh

```bash
chmod +x scripts/setup_cron.sh
./scripts/setup_cron.sh
```

This sets up a cron job to refresh the index every night at 2 AM.

### Manual Refresh

```bash
python scripts/refresh_index.py
```

## 📁 Project Structure

```
QueryGenie/
├── backend/                    # FastAPI backend application
│   ├── api/
│   │   └── v1/
│   │       └── routes.py      # API v1 endpoints
│   └── main.py                # Backend entry point
├── frontend/                   # React + TypeScript frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── services/          # API services
│   │   └── App.tsx            # Main app component
│   └── Dockerfile             # Frontend container
├── src/                        # Core RAG logic (shared)
│   ├── __init__.py
│   ├── api.py                 # Legacy API (deprecated)
│   ├── arxiv_downloader.py    # Paper downloader
│   ├── preprocessing.py       # Document processing
│   ├── faiss_manager.py       # FAISS index management
│   ├── rag_pipeline.py        # RAG pipeline
│   └── llm_generator.py       # LLM generation
├── scripts/
│   ├── refresh_index.py       # Index refresh script
│   └── setup_cron.sh          # Cron job setup
├── data/                       # Data directory (FAISS index, papers)
├── models/                     # LLM model files (if using LLM)
├── requirements.txt            # Python dependencies
├── Dockerfile.backend         # Backend Docker configuration
├── docker-compose.prod.yml    # Production Docker Compose setup
├── test_queries.json          # Test queries
└── README.md                  # This file
```

## 🐳 Docker Deployment

### Production Deployment (Recommended)

```bash
# Start all services (backend + frontend)
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop services
docker-compose -f docker-compose.prod.yml down
```

The services will be available at:

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Manual Docker Build

```bash
# Build backend image
docker build -f Dockerfile.backend -t querygenie-backend .

# Run backend container
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -e USE_LLM=true \
  querygenie-backend
```

## 🔍 Troubleshooting

### Common Issues

1. **"FAISS index not found"**

   - Run the preprocessing pipeline first
   - Check if `data/faiss_index.faiss` exists

2. **"Out of memory"**

   - Reduce `chunk_size` in preprocessing
   - Use a smaller embedding model
   - Close other applications

3. **"Model download failed"**

   - Check internet connection
   - Clear Hugging Face cache: `rm -rf ~/.cache/huggingface`

4. **"Slow performance"**
   - Use GPU if available (set `device="cuda"`)
   - Reduce `max_context_length`
   - Use fewer retrieved sources

### Performance Optimization

- **GPU Acceleration**: Set `device="cuda"` in RAGPipeline
- **Memory Usage**: Adjust `chunk_size` and `batch_size`
- **Index Size**: Limit number of papers downloaded

## 📈 Performance Benchmarks

On MacBook Air M2 (8GB RAM):

- **Index Creation**: ~5-10 minutes for 200 papers
- **Query Response**: ~2-5 seconds per query
- **Memory Usage**: ~2-4GB during operation
- **Index Size**: ~100-500MB depending on corpus size

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Hugging Face** for providing free, open-source models
- **Facebook AI** for FAISS similarity search
- **arXiv** for providing open access to research papers
- **FastAPI** for the excellent web framework

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs in `logs/` directory
3. Open an issue on GitHub
4. Check the API documentation at `http://localhost:8000/docs` (interactive Swagger UI)

---

**QueryGenie** - Bringing the power of RAG to your local machine, completely free! 🚀
