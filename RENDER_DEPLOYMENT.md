# Render Deployment Guide for QueryGenie

Complete step-by-step guide to deploy QueryGenie to Render's free tier.

## 📋 Prerequisites

- [ ] Render account (sign up at [render.com](https://render.com) - free tier available)
- [ ] GitHub account with your QueryGenie repository
- [ ] FAISS index built locally (recommended) or ready to build on first deploy
- [ ] Docker installed locally (for testing, optional)

## 🎯 Overview

We'll deploy:
1. **Backend**: Web Service (Docker) - FastAPI application
2. **Frontend**: Static Site - React application
3. **Storage**: Render Disk - Persistent storage for FAISS index and models

**Estimated Time**: 20-30 minutes  
**Cost**: Free (within free tier limits)

---

## Step 1: Prepare Your Repository

### 1.1 Ensure Code is Committed

```bash
git status
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 1.2 Verify FAISS Index Exists

```bash
# Check if index exists locally
ls -lh data/faiss_index.faiss
ls -lh data/faiss_index_chunks.json
```

**Important**: You'll need to upload the FAISS index to Render Disk after deployment, or build it on first deploy.

---

## Step 2: Create Backend Service

### 2.1 Create New Web Service

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Select your QueryGenie repository

### 2.2 Configure Backend Service

**Basic Settings:**
- **Name**: `querygenie-backend`
- **Region**: Choose closest to you (Oregon, Frankfurt, Singapore)
- **Branch**: `main` (or your default branch)
- **Root Directory**: Leave empty (root of repo)
- **Environment**: Select **"Docker"**
- **Dockerfile Path**: `Dockerfile.backend`
- **Docker Context**: `.` (dot)
- **Plan**: **Free**

**Advanced Settings:**
- **Health Check Path**: `/api/v1/health`
- **Auto-Deploy**: `Yes` (deploys on git push)

### 2.3 Add Environment Variables

Click **"Environment"** tab and add:

```bash
USE_LLM=true
HOST=0.0.0.0
PORT=8000
RELOAD=false
ALLOWED_HOSTS=querygenie-backend.onrender.com
```

**Note**: We'll add `CORS_ORIGINS` after frontend deploys.

### 2.4 Add Persistent Disks

Click **"Disks"** tab and add:

**Disk 1:**
- **Name**: `querygenie-data`
- **Mount Path**: `/app/data`
- **Size**: 1 GB (free tier limit)

**Disk 2 (if using LLM):**
- **Name**: `querygenie-models`
- **Mount Path**: `/app/models`
- **Size**: 1 GB

### 2.5 Deploy Backend

1. Click **"Create Web Service"**
2. Wait for build to complete (5-10 minutes first time)
3. Note your backend URL (e.g., `https://querygenie-backend.onrender.com`)

### 2.6 Verify Backend

```bash
# Test health endpoint
curl https://your-backend.onrender.com/api/v1/health

# Should return:
# {"status":"healthy" or "degraded","pipeline_ready":true/false,"timestamp":"..."}
```

**If pipeline_ready is false**: FAISS index not found. We'll upload it in Step 4.

---

## Step 3: Create Frontend Service

### 3.1 Create New Static Site

1. In Render Dashboard, click **"New +"** → **"Static Site"**
2. Connect your GitHub repository (same one)
3. Select your QueryGenie repository

### 3.2 Configure Frontend Service

**Basic Settings:**
- **Name**: `querygenie-frontend`
- **Branch**: `main`
- **Root Directory**: Leave empty
- **Build Command**: `cd frontend && npm install && npm run build`
- **Publish Directory**: `frontend/dist`
- **Plan**: **Free**

### 3.3 Add Environment Variable

**Critical**: Add this before first build:

```bash
VITE_API_URL=https://your-backend.onrender.com/api
```

Replace `your-backend` with your actual backend service name.

**Important**: This must be set **before** the first build. Vite embeds this at build time.

### 3.4 Deploy Frontend

1. Click **"Create Static Site"**
2. Wait for build (2-5 minutes)
3. Note your frontend URL (e.g., `https://querygenie-frontend.onrender.com`)

---

## Step 4: Update Backend CORS

### 4.1 Add Frontend URL to CORS

1. Go to your backend service in Render Dashboard
2. Click **"Environment"** tab
3. Add/Update environment variable:

```bash
CORS_ORIGINS=https://your-frontend.onrender.com
```

Replace `your-frontend` with your actual frontend service name.

4. Backend will automatically restart

### 4.2 Verify CORS

```bash
# Test from browser console on frontend
fetch('https://your-backend.onrender.com/api/v1/health')
  .then(r => r.json())
  .then(console.log)
```

Should work without CORS errors.

---

## Step 5: Upload FAISS Index (If Pre-built)

### Option A: Upload via Render Shell

1. Go to backend service → **"Shell"** tab
2. Upload files:

```bash
# Create data directory if needed
mkdir -p /app/data

# Upload faiss_index.faiss and faiss_index_chunks.json
# Use Render's file upload or SCP
```

### Option B: Build Index on First Deploy

1. Go to backend service → **"Shell"** tab
2. Run:

```bash
cd /app
python src/arxiv_downloader.py
python src/preprocessing.py
```

**Note**: This may take 10-30 minutes depending on number of papers.

### Option C: Include in Docker Image (If Small)

If your index is < 500MB, you can include it in the Docker image:

```dockerfile
# Add to Dockerfile.backend before CMD
COPY data/faiss_index.faiss ./data/
COPY data/faiss_index_chunks.json ./data/
```

---

## Step 6: Test Deployment

### 6.1 Test Backend

```bash
# Health check
curl https://your-backend.onrender.com/api/v1/health

# Metrics
curl https://your-backend.onrender.com/api/v1/metrics

# API docs
open https://your-backend.onrender.com/docs
```

### 6.2 Test Frontend

1. Open frontend URL: `https://your-frontend.onrender.com`
2. Check browser console for errors
3. Try asking a question
4. Verify API calls work (check Network tab)

### 6.3 Test End-to-End

1. Open frontend in browser
2. Ask a test question: "What are transformers?"
3. Verify:
   - Question is sent to backend
   - Answer is received
   - Sources are displayed
   - No CORS errors in console

---

## 🔧 Configuration Reference

See [RENDER_ENV_VARS.md](RENDER_ENV_VARS.md) for complete environment variable reference.

---

## 🚨 Troubleshooting

### Backend Won't Start

**Issue**: Service fails to start

**Solutions**:
- Check logs in Render Dashboard
- Verify `PORT` environment variable (Render sets automatically)
- Ensure `HOST=0.0.0.0`
- Check if FAISS index exists (service will start but pipeline won't be ready)

### Frontend Can't Connect to Backend

**Issue**: CORS errors or connection refused

**Solutions**:
- Verify `VITE_API_URL` is set correctly (must include `/api`)
- Check backend URL is correct
- Ensure `CORS_ORIGINS` includes frontend URL
- Rebuild frontend if `VITE_API_URL` was changed

### FAISS Index Not Found

**Issue**: `pipeline_ready: false` in health check

**Solutions**:
- Upload index files to Render Disk at `/app/data`
- Or build index via Shell (see Step 5)
- Verify disk is mounted correctly

### Out of Memory

**Issue**: Service crashes or runs out of memory

**Solutions**:
- Set `USE_LLM=false` (retrieval-only mode)
- Reduce FAISS index size
- Use smaller embedding model
- Consider upgrading plan (not free)

### Cold Starts

**Issue**: First request takes 30-60 seconds

**Solutions**:
- This is normal for Render free tier
- Services spin down after 15 min inactivity
- Consider upgrading to paid plan for always-on

---

## 📊 Free Tier Limitations

**Important**: Render free tier has limitations:

- **Memory**: 512MB RAM (may not be enough with LLM)
- **CPU**: Shared resources
- **Disk**: 1GB per disk (2 disks = 2GB total)
- **Spins Down**: After 15 minutes of inactivity
- **Cold Start**: 30-60 seconds after spin-down

**Recommendations for Free Tier**:
- Use `USE_LLM=false` (retrieval-only mode)
- Keep FAISS index small (100-200 papers)
- Accept cold starts (normal for portfolio)

---

## 🔄 Updating Deployment

### Automatic Updates

Render automatically deploys when you push to your connected branch:

```bash
git push origin main
```

### Manual Updates

1. Go to service in Render Dashboard
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

### Environment Variable Updates

1. Go to service → **"Environment"** tab
2. Add/Update variables
3. Service restarts automatically

---

## 📈 Monitoring

### View Logs

1. Go to service in Render Dashboard
2. Click **"Logs"** tab
3. Real-time logs are available

### Health Checks

Render automatically checks `/api/v1/health` every 30 seconds.

### Metrics

Access metrics endpoint:
```bash
curl https://your-backend.onrender.com/api/v1/metrics
```

---

## 🎉 Success Checklist

- [ ] Backend service deployed and healthy
- [ ] Frontend service deployed and accessible
- [ ] CORS configured correctly
- [ ] FAISS index uploaded/built
- [ ] Frontend can query backend
- [ ] No errors in browser console
- [ ] Health endpoint returns `pipeline_ready: true`

---

## 🔗 Quick Links

After deployment, your services will be at:

- **Backend**: `https://querygenie-backend.onrender.com`
- **Frontend**: `https://querygenie-frontend.onrender.com`
- **API Docs**: `https://querygenie-backend.onrender.com/docs`

---

## 📚 Additional Resources

- [Render Documentation](https://render.com/docs)
- [Environment Variables Reference](RENDER_ENV_VARS.md)
- [Troubleshooting Guide](RENDER_DEPLOYMENT.md#-troubleshooting)

---

**Need Help?** Check the troubleshooting section or Render's support documentation.

