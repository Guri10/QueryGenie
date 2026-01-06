# Render Environment Variables Reference

Complete reference for all environment variables needed for QueryGenie deployment on Render.

## Backend Service Environment Variables

### Required Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `USE_LLM` | `true` or `false` | Enable/disable LLM generation. Set to `false` for free tier to save memory. |
| `HOST` | `0.0.0.0` | Server host (required for Render) |
| `PORT` | `8000` | Server port (Render sets this automatically, but default is 8000) |
| `RELOAD` | `false` | Disable auto-reload in production |

### Optional Variables

| Variable | Value | Description | When to Set |
|----------|-------|-------------|-------------|
| `CORS_ORIGINS` | Comma-separated URLs | Allowed CORS origins | **After frontend deploys** - Set to your frontend URL (e.g., `https://querygenie-frontend.onrender.com`) |
| `ALLOWED_HOSTS` | Comma-separated hosts | Trusted hosts for security | Set to your backend domain (e.g., `querygenie-backend.onrender.com`) |
| `LLM_MODEL` | Model name | LLM model to use | Only if `USE_LLM=true`. Options: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `microsoft/phi-2`, `mistralai/Mistral-7B-Instruct-v0.2` |

### Example Backend Configuration

```bash
USE_LLM=true
HOST=0.0.0.0
PORT=8000
RELOAD=false
ALLOWED_HOSTS=querygenie-backend.onrender.com
CORS_ORIGINS=https://querygenie-frontend.onrender.com
LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0  # Optional
```

---

## Frontend Service Environment Variables

### Required Variables

| Variable | Value | Description | When to Set |
|----------|-------|-------------|-------------|
| `VITE_API_URL` | Backend API URL | Full URL to backend API | **After backend deploys** - Set to `https://your-backend.onrender.com/api` |

### Example Frontend Configuration

```bash
VITE_API_URL=https://querygenie-backend.onrender.com/api
```

**Important:** `VITE_API_URL` must be set **before building** the frontend. Vite embeds this at build time, not runtime.

---

## Deployment Order

Because some variables depend on service URLs, deploy in this order:

1. **Deploy Backend First**
   - Set: `USE_LLM`, `HOST`, `PORT`, `RELOAD`, `ALLOWED_HOSTS`
   - Note the backend URL (e.g., `https://querygenie-backend.onrender.com`)

2. **Deploy Frontend**
   - Set: `VITE_API_URL=https://your-backend.onrender.com/api`
   - Build will use this URL

3. **Update Backend CORS**
   - Update `CORS_ORIGINS` with frontend URL (e.g., `https://querygenie-frontend.onrender.com`)
   - Backend will restart automatically

---

## Free Tier Optimizations

To stay within Render's free tier limits:

```bash
# Backend
USE_LLM=false  # Disable LLM to save memory (512MB limit)
HOST=0.0.0.0
PORT=8000
RELOAD=false
ALLOWED_HOSTS=querygenie-backend.onrender.com
CORS_ORIGINS=https://querygenie-frontend.onrender.com
```

This runs in **retrieval-only mode** (no LLM generation), which:
- Uses less memory (~200-400MB vs 1-4GB+)
- Faster startup
- Still provides full RAG functionality (retrieval + formatted context)

---

## Environment Variable Format

### Comma-Separated Values

For `CORS_ORIGINS` and `ALLOWED_HOSTS`, use comma-separated values:

```bash
# Multiple origins
CORS_ORIGINS=https://frontend1.onrender.com,https://frontend2.onrender.com

# Multiple hosts
ALLOWED_HOSTS=backend1.onrender.com,backend2.onrender.com
```

Spaces are automatically trimmed, but avoid them for clarity.

---

## Troubleshooting

### Frontend can't connect to backend
- Check `VITE_API_URL` is set correctly
- Ensure backend URL includes `/api` suffix
- Verify backend is running (check health endpoint)

### CORS errors
- Verify `CORS_ORIGINS` includes your frontend URL
- Check for typos in URLs (http vs https, trailing slashes)
- Backend must restart after changing `CORS_ORIGINS`

### Backend won't start
- Check `PORT` is set (Render sets this automatically)
- Verify `HOST=0.0.0.0` (required for Render)
- Check logs for specific errors

### Out of memory
- Set `USE_LLM=false` for free tier
- Reduce FAISS index size (fewer papers)
- Use smaller embedding model

---

## Quick Reference

**Backend Minimum:**
```
USE_LLM=false
HOST=0.0.0.0
PORT=8000
RELOAD=false
```

**Frontend Minimum:**
```
VITE_API_URL=https://your-backend.onrender.com/api
```

**After Both Deploy:**
```
# Backend - add CORS
CORS_ORIGINS=https://your-frontend.onrender.com
ALLOWED_HOSTS=your-backend.onrender.com
```

