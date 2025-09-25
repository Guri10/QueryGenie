# QueryGenie RAG Chatbot

A **completely free, local-only** Retrieval-Augmented Generation (RAG) chatbot that uses arXiv papers as its knowledge base. Built with open-source components and designed to run entirely on your local machine with zero costs.

## 🚀 Features

- **Zero Cost**: Uses only free, open-source models and libraries
- **Local Only**: Runs entirely on your machine, no external APIs
- **arXiv Integration**: Automatically downloads and indexes recent research papers
- **Fast Retrieval**: FAISS-based similarity search for efficient document retrieval
- **Smart Retrieval**: Intelligent document retrieval with similarity scoring
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
   python src/api.py
   ```

### Option 2: Docker Deployment

1. **Build and run with Docker Compose**

   ```bash
   docker-compose up -d
   ```

2. **Initialize the system** (first time only)
   ```bash
   # Download papers and create index
   docker-compose exec querygenie python src/arxiv_downloader.py
   docker-compose exec querygenie python src/preprocessing.py
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
python src/api.py
```

The API will be available at `http://localhost:8000`

### 4. Query the System

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the latest advances in transformer architectures?"}'
```

## 📚 API Endpoints

### Core Endpoints

- **`POST /ask`** - Ask a question to the RAG system
- **`GET /health`** - Health check and system status
- **`GET /metrics`** - Performance metrics and statistics
- **`POST /refresh`** - Trigger index refresh
- **`GET /stats`** - Detailed system statistics

### Example Usage

#### Ask a Question

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "How do attention mechanisms work in transformers?",
       "k": 5,
       "max_context_length": 1000,
       "max_answer_length": 200
     }'
```

#### Check System Health

```bash
curl http://localhost:8000/health
```

#### Get Performance Metrics

```bash
curl http://localhost:8000/metrics
```

## 🔧 Configuration

### Model Configuration

The system uses these free, open-source models:

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, ~90MB)
- **Retrieval**: FAISS-based similarity search with sentence-transformers

### Customization

You can modify the models in the source code:

```python
# In src/preprocessing.py
processor = DocumentProcessor(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Change this
    chunk_size=512,
    chunk_overlap=50
)

# In src/rag_pipeline.py
rag_pipeline = RAGPipeline(
    faiss_manager,
    generator_model="distilgpt2"  # Change this
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
├── src/
│   ├── __init__.py
│   ├── api.py                 # FastAPI service
│   ├── arxiv_downloader.py    # Paper downloader
│   ├── preprocessing.py       # Document processing
│   ├── faiss_manager.py       # FAISS index management
│   ├── rag_pipeline.py        # RAG pipeline
│   └── evaluation.py          # Evaluation scripts
├── scripts/
│   ├── refresh_index.py       # Index refresh script
│   └── setup_cron.sh          # Cron job setup
├── data/                       # Data directory
├── logs/                       # Log files
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── test_queries.json          # Test queries
└── README.md                  # This file
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t querygenie .

# Run the container
docker run -p 8000:8000 -v $(pwd)/data:/app/data querygenie
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
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
4. Check the API documentation at `http://localhost:8000/docs`

---

**QueryGenie** - Bringing the power of RAG to your local machine, completely free! 🚀
