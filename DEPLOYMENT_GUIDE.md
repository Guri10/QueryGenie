# QueryGenie Deployment Guide

This guide covers different deployment options for the QueryGenie frontend-backend architecture.

## 🚀 Quick Deployment Options

### 1. Local Development

```bash
# Run the setup script
./setup_frontend_backend.sh

# Start everything
./start_dev.sh
```

### 2. Docker Production

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

## ☁️ Cloud Deployment

### Heroku

1. **Install Heroku CLI**
2. **Create Heroku apps:**

   ```bash
   # Backend
   heroku create querygenie-backend

   # Frontend
   heroku create querygenie-frontend
   ```

3. **Deploy Backend:**

   ```bash
   # Add buildpack
   heroku buildpacks:set heroku/python -a querygenie-backend

   # Set environment variables
   heroku config:set USE_LLM=true -a querygenie-backend
   heroku config:set HOST=0.0.0.0 -a querygenie-backend
   heroku config:set PORT=8000 -a querygenie-backend

   # Deploy
   git subtree push --prefix=backend heroku main
   ```

4. **Deploy Frontend:**

   ```bash
   # Add buildpack
   heroku buildpacks:set heroku/nodejs -a querygenie-frontend

   # Set environment variables
   heroku config:set VITE_API_URL=https://querygenie-backend.herokuapp.com/api -a querygenie-frontend

   # Deploy
   git subtree push --prefix=frontend heroku main
   ```

### AWS (ECS/Fargate)

1. **Create ECR repositories:**

   ```bash
   aws ecr create-repository --repository-name querygenie-backend
   aws ecr create-repository --repository-name querygenie-frontend
   ```

2. **Build and push images:**

   ```bash
   # Backend
   docker build -f Dockerfile.backend -t querygenie-backend .
   docker tag querygenie-backend:latest $ECR_REGISTRY/querygenie-backend:latest
   docker push $ECR_REGISTRY/querygenie-backend:latest

   # Frontend
   docker build -f frontend/Dockerfile -t querygenie-frontend ./frontend
   docker tag querygenie-frontend:latest $ECR_REGISTRY/querygenie-frontend:latest
   docker push $ECR_REGISTRY/querygenie-frontend:latest
   ```

3. **Create ECS task definitions and services**

### Google Cloud (Cloud Run)

1. **Deploy Backend:**

   ```bash
   gcloud run deploy querygenie-backend \
     --source=backend \
     --platform=managed \
     --region=us-central1 \
     --allow-unauthenticated
   ```

2. **Deploy Frontend:**
   ```bash
   gcloud run deploy querygenie-frontend \
     --source=frontend \
     --platform=managed \
     --region=us-central1 \
     --allow-unauthenticated
   ```

### Azure (Container Instances)

1. **Create resource group:**

   ```bash
   az group create --name querygenie-rg --location eastus
   ```

2. **Deploy containers:**

   ```bash
   # Backend
   az container create \
     --resource-group querygenie-rg \
     --name querygenie-backend \
     --image your-registry/querygenie-backend:latest \
     --ports 8000

   # Frontend
   az container create \
     --resource-group querygenie-rg \
     --name querygenie-frontend \
     --image your-registry/querygenie-frontend:latest \
     --ports 80
   ```

## 🐳 Docker Deployment

### Single Container (All-in-One)

```bash
# Build the image
docker build -t querygenie .

# Run the container
docker run -p 8000:8000 -p 3000:3000 querygenie
```

### Multi-Container (Recommended)

```bash
# Using Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
```

## 🔧 Environment Configuration

### Required Environment Variables

**Backend:**

```bash
USE_LLM=true
LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
HOST=0.0.0.0
PORT=8000
RELOAD=false
```

**Frontend:**

```bash
VITE_API_URL=http://localhost:8000/api
```

### Optional Environment Variables

```bash
# Database (if using external DB)
DATABASE_URL=postgresql://user:password@localhost:5432/querygenie

# Redis (if using caching)
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# Monitoring
SENTRY_DSN=your-sentry-dsn-here
```

## 📊 Monitoring & Health Checks

### Health Check Endpoints

- **Backend Health:** `GET /api/v1/health`
- **Backend Metrics:** `GET /api/v1/metrics`

### Docker Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1
```

### Monitoring Setup

1. **Prometheus + Grafana** for metrics
2. **ELK Stack** for logging
3. **Sentry** for error tracking

## 🔒 Security Considerations

### Production Security Checklist

- [ ] Enable HTTPS/TLS
- [ ] Set up proper CORS origins
- [ ] Configure trusted hosts
- [ ] Use environment variables for secrets
- [ ] Enable rate limiting
- [ ] Set up authentication (if needed)
- [ ] Regular security updates

### SSL/TLS Setup

```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    # ... rest of configuration
}
```

## 📈 Scaling

### Horizontal Scaling

- **Load Balancer** (nginx, HAProxy)
- **Multiple Backend Instances**
- **CDN** for frontend assets
- **Database Clustering** (if using external DB)

### Vertical Scaling

- **Increase Memory** for larger FAISS indexes
- **More CPU Cores** for parallel processing
- **SSD Storage** for faster I/O

## 🚨 Troubleshooting

### Common Issues

1. **CORS Errors:**

   - Check CORS configuration in `backend/main.py`
   - Verify frontend URL in allowed origins

2. **API Connection Issues:**

   - Check `VITE_API_URL` environment variable
   - Verify backend is running on correct port

3. **Memory Issues:**

   - Increase container memory limits
   - Optimize FAISS index size

4. **Build Failures:**
   - Check Node.js and Python versions
   - Verify all dependencies are installed

### Debug Commands

```bash
# Check container logs
docker logs querygenie-backend
docker logs querygenie-frontend

# Check container status
docker ps -a

# Check resource usage
docker stats

# Access container shell
docker exec -it querygenie-backend bash
```

## 📚 Additional Resources

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [React Deployment](https://create-react-app.dev/docs/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Nginx Configuration](https://nginx.org/en/docs/)

---

**Happy Deploying!** 🚀
