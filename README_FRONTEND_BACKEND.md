# QueryGenie - Frontend & Backend Architecture

A modern, production-ready RAG chatbot with separate frontend and backend components.

## 🏗️ Architecture Overview

```
QueryGenie/
├── frontend/                 # React + TypeScript frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── services/        # API services
│   │   ├── types/          # TypeScript types
│   │   └── App.tsx         # Main app component
│   ├── public/             # Static assets
│   ├── package.json        # Frontend dependencies
│   └── Dockerfile          # Frontend container
├── backend/                 # FastAPI backend
│   ├── api/
│   │   └── v1/            # API version 1
│   │       └── routes.py   # API endpoints
│   └── main.py            # Backend entry point
├── src/                    # Core RAG logic (shared)
│   ├── api.py             # Original API (legacy)
│   ├── rag_pipeline.py    # RAG pipeline
│   ├── faiss_manager.py   # FAISS management
│   └── llm_generator.py   # LLM generation
└── docker-compose.prod.yml # Production deployment
```

## 🚀 Quick Start

### Development Mode

1. **Start Backend:**

   ```bash
   cd backend
   python main.py
   ```

2. **Start Frontend:**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Access Application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Production Mode

1. **Using Docker Compose:**

   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **Access Application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## 🎯 Features

### Frontend Features

- ✅ **Modern React UI** with TypeScript
- ✅ **Real-time Chat Interface** with message history
- ✅ **Source Citations** with expandable details
- ✅ **Performance Metrics** display
- ✅ **Responsive Design** with Tailwind CSS
- ✅ **Loading States** and error handling
- ✅ **System Statistics** dashboard

### Backend Features

- ✅ **RESTful API** with FastAPI
- ✅ **API Versioning** (v1)
- ✅ **CORS Support** for frontend integration
- ✅ **Health Checks** and monitoring
- ✅ **Error Handling** with proper HTTP status codes
- ✅ **Background Tasks** for index refresh
- ✅ **Production Ready** with proper logging

### Core RAG Features

- ✅ **10,663 Research Papers** (20 years of CS history)
- ✅ **Smart Retrieval** with FAISS optimization
- ✅ **LLM Generation** with grounding validation
- ✅ **Citation Tracking** with similarity scores
- ✅ **Multi-domain Coverage** (AI, ML, CV, NLP, etc.)

## 🛠️ Development

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint

# Type check
npm run type-check
```

### Backend Development

```bash
cd backend

# Install dependencies (from project root)
pip install -r requirements.txt

# Start development server
python main.py

# Or with auto-reload
RELOAD=true python main.py
```

## 📡 API Endpoints

### Health & Monitoring

- `GET /api/v1/health` - Health check
- `GET /api/v1/metrics` - System metrics

### Core Functionality

- `POST /api/v1/ask` - Ask a question
- `POST /api/v1/refresh` - Refresh index

### Example API Usage

```bash
# Ask a question
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are transformer architectures?",
    "k": 5,
    "max_context_length": 5000,
    "max_answer_length": 300
  }'

# Check health
curl http://localhost:8000/api/v1/health

# Get metrics
curl http://localhost:8000/api/v1/metrics
```

## 🐳 Deployment

### Docker Deployment

1. **Build and run:**

   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **View logs:**

   ```bash
   docker-compose -f docker-compose.prod.yml logs -f
   ```

3. **Stop services:**
   ```bash
   docker-compose -f docker-compose.prod.yml down
   ```

### Cloud Deployment

#### Heroku

```bash
# Install Heroku CLI
# Create Procfile
echo "web: python backend/main.py" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

#### AWS/GCP/Azure

- Use the provided Dockerfiles
- Deploy to container services (ECS, Cloud Run, Container Instances)
- Configure load balancers and auto-scaling

## 🔧 Configuration

### Environment Variables

Copy `env.example` to `.env` and configure:

```bash
# Backend
USE_LLM=true
LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
HOST=0.0.0.0
PORT=8000

# Frontend
VITE_API_URL=http://localhost:8000/api
```

### Frontend Configuration

Edit `frontend/vite.config.js` for build settings:

- API proxy configuration
- Build output directory
- Source maps

### Backend Configuration

Edit `backend/main.py` for:

- CORS origins
- Trusted hosts
- Logging levels

## 📊 Monitoring

### Health Checks

- Backend: `GET /api/v1/health`
- Frontend: Built-in error boundaries

### Metrics

- `GET /api/v1/metrics` - System statistics
- Paper count, index size, performance metrics

### Logging

- Structured logging with timestamps
- Error tracking and debugging
- Production-ready log levels

## 🚀 Performance

### Frontend Optimizations

- ✅ **Code Splitting** with Vite
- ✅ **Tree Shaking** for smaller bundles
- ✅ **Gzip Compression** in production
- ✅ **Static Asset Caching**
- ✅ **Lazy Loading** of components

### Backend Optimizations

- ✅ **Async/Await** for non-blocking I/O
- ✅ **Connection Pooling** for database
- ✅ **Response Caching** for static data
- ✅ **Background Tasks** for heavy operations

### RAG Optimizations

- ✅ **FAISS IVF Index** for fast retrieval
- ✅ **Query Caching** for repeated questions
- ✅ **Batch Processing** for multiple queries
- ✅ **LLM Grounding** to prevent hallucination

## 🔒 Security

### Frontend Security

- ✅ **CSP Headers** for XSS protection
- ✅ **Input Validation** and sanitization
- ✅ **HTTPS Enforcement** in production
- ✅ **Secure API Communication**

### Backend Security

- ✅ **CORS Configuration** for cross-origin requests
- ✅ **Input Validation** with Pydantic
- ✅ **Error Handling** without information leakage
- ✅ **Rate Limiting** (can be added)

## 🧪 Testing

### Frontend Testing

```bash
cd frontend
npm run test        # Unit tests
npm run test:e2e    # End-to-end tests
npm run coverage    # Coverage report
```

### Backend Testing

```bash
cd backend
python -m pytest tests/    # Unit tests
python -m pytest tests/ --cov  # With coverage
```

## 📈 Scaling

### Horizontal Scaling

- **Frontend**: CDN + multiple instances
- **Backend**: Load balancer + multiple API instances
- **Database**: Read replicas + connection pooling

### Vertical Scaling

- **Memory**: Increase for larger FAISS indexes
- **CPU**: More cores for parallel processing
- **Storage**: SSD for faster I/O

## 🎯 Next Steps

1. **Add Authentication** (JWT, OAuth)
2. **Implement Caching** (Redis)
3. **Add Database** (PostgreSQL)
4. **Set up Monitoring** (Prometheus, Grafana)
5. **Add CI/CD** (GitHub Actions)
6. **Implement Rate Limiting**
7. **Add User Management**
8. **Create Admin Dashboard**

## 📚 Documentation

- **API Docs**: http://localhost:8000/docs
- **Frontend Components**: See `frontend/src/components/`
- **Backend Routes**: See `backend/api/v1/routes.py`
- **Core Logic**: See `src/` directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**QueryGenie** - Your AI-powered research assistant! 🚀
