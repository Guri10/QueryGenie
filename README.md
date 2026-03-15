# QueryGenie

An AI-powered research assistant that answers questions about arXiv papers using Retrieval-Augmented Generation (RAG). Ask anything about recent ML/AI research and get answers synthesized from real papers.

**Live:** [querygenie-frontend-wkk3.onrender.com](https://querygenie-frontend-wkk3.onrender.com)

---

## Architecture

```
arXiv Papers → Preprocessing → FAISS Index (sentence-transformers embeddings)
                                      ↓
User Query → RAG Pipeline → OpenAI gpt-4o-mini → Answer + Sources
                                      ↓
                              FastAPI Backend ← React Frontend
```

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384 dims, 24k chunks indexed)
- **Vector Store:** FAISS (local, committed to repo for zero-cost deployment)
- **LLM:** OpenAI `gpt-4o-mini` via API
- **Backend:** FastAPI (Python 3.11, Docker)
- **Frontend:** React 18 + TypeScript + Tailwind CSS + Vite 5
- **Hosting:** Render.com (backend: Docker web service, frontend: static site)

---

## Local Development

### Prerequisites
- Python 3.9+
- Node.js 18+
- OpenAI API key

### Backend

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create a .env file
echo "OPENAI_API_KEY=sk-..." > .env
echo "USE_LLM=true" >> .env
echo "LLM_PROVIDER=openai" >> .env
echo "LLM_MODEL=gpt-4o-mini" >> .env

cd backend
export PYTHONPATH="../src:$PYTHONPATH"
python main.py
```

Backend runs at `http://localhost:8000` · Swagger docs at `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ask` | Ask a question |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/metrics` | Index stats |

**Example:**
```bash
curl -X POST https://querygenie-backend-7l0g.onrender.com/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is attention mechanism in transformers?", "k": 3}'
```

---

## Deployment (Render.com)

### Backend — Docker Web Service

| Variable | Value |
|----------|-------|
| `USE_LLM` | `true` |
| `LLM_PROVIDER` | `openai` |
| `LLM_MODEL` | `gpt-4o-mini` |
| `OPENAI_API_KEY` | your key |
| `CORS_ORIGINS` | frontend URL |
| `HOST` | `0.0.0.0` |
| `PORT` | `8000` |

### Frontend — Static Site

| Variable | Value |
|----------|-------|
| `VITE_API_URL` | `https://querygenie-backend-7l0g.onrender.com/api/v1` |

Build command: `cd frontend && rm -rf node_modules && npm install && npm run build`  
Publish directory: `frontend/dist`

---

## Project Structure

```
QueryGenie/
├── backend/
│   ├── api/v1/routes.py      # API endpoints
│   └── main.py               # FastAPI app entry point
├── frontend/
│   └── src/
│       ├── components/        # React components
│       ├── services/          # API client
│       └── App.tsx
├── src/                       # Core RAG logic
│   ├── faiss_manager.py       # FAISS index management
│   ├── rag_pipeline.py        # Retrieval + generation
│   ├── llm_generator.py       # Local & OpenAI LLM backends
│   ├── arxiv_downloader.py    # Paper downloader
│   └── preprocessing.py       # Chunking + embedding
├── data/
│   ├── faiss_index.faiss      # Pre-built vector index
│   └── faiss_index_chunks.json
├── Dockerfile.backend
├── render.yaml
└── requirements.txt
```

---

## License

MIT
