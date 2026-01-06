#!/bin/bash
# Pre-deployment checklist script for Render
# Verifies that everything is ready before deploying to Render

set -e

echo "🔍 QueryGenie Render Deployment Preparation"
echo "==========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅${NC} $2"
        return 0
    else
        echo -e "${RED}❌${NC} $2 (missing: $1)"
        ((ERRORS++))
        return 1
    fi
}

# Function to check directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✅${NC} $2"
        return 0
    else
        echo -e "${RED}❌${NC} $2 (missing: $1)"
        ((ERRORS++))
        return 1
    fi
}

# Function to warn
warn() {
    echo -e "${YELLOW}⚠️${NC} $1"
    ((WARNINGS++))
}

echo "1️⃣  Checking Required Files"
echo "----------------------------"

check_file "Dockerfile.backend" "Backend Dockerfile"
check_file "frontend/Dockerfile" "Frontend Dockerfile"
check_file "frontend/package.json" "Frontend package.json"
check_file "requirements.txt" "Python requirements"
check_file "backend/main.py" "Backend main file"
check_file "backend/api/v1/routes.py" "Backend API routes"

echo ""
echo "2️⃣  Checking Code Configuration"
echo "-------------------------------"

# Check CORS configuration
if grep -q "CORS_ORIGINS" backend/main.py; then
    echo -e "${GREEN}✅${NC} CORS environment variable support"
else
    echo -e "${RED}❌${NC} CORS environment variable support missing"
    ((ERRORS++))
fi

# Check Trusted Hosts configuration
if grep -q "ALLOWED_HOSTS" backend/main.py; then
    echo -e "${GREEN}✅${NC} Trusted hosts environment variable support"
else
    echo -e "${RED}❌${NC} Trusted hosts environment variable support missing"
    ((ERRORS++))
fi

# Check Dockerfile USE_LLM
if grep -q "ENV USE_LLM" Dockerfile.backend; then
    echo -e "${GREEN}✅${NC} Dockerfile has USE_LLM configured"
else
    echo -e "${RED}❌${NC} Dockerfile missing USE_LLM"
    ((ERRORS++))
fi

echo ""
echo "3️⃣  Checking Data Files"
echo "-----------------------"

if [ -f "data/faiss_index.faiss" ]; then
    SIZE=$(du -h data/faiss_index.faiss | cut -f1)
    echo -e "${GREEN}✅${NC} FAISS index exists (size: $SIZE)"
    
    if [ -f "data/faiss_index_chunks.json" ]; then
        echo -e "${GREEN}✅${NC} FAISS chunks metadata exists"
    else
        warn "FAISS chunks metadata missing (will need to rebuild index)"
    fi
else
    warn "FAISS index not found - you'll need to build it on Render or upload it"
fi

# Check if data directory has files
if [ -d "data" ] && [ "$(ls -A data/*.json 2>/dev/null)" ]; then
    PAPER_COUNT=$(ls -1 data/arxiv_papers_*.json 2>/dev/null | wc -l)
    echo -e "${BLUE}ℹ️${NC}  Found $PAPER_COUNT paper JSON file(s)"
else
    warn "No paper data files found in data/ directory"
fi

echo ""
echo "4️⃣  Checking Git Status"
echo "----------------------"

if command -v git &> /dev/null; then
    if [ -d ".git" ]; then
        # Check if there are uncommitted changes
        if [ -n "$(git status --porcelain)" ]; then
            warn "You have uncommitted changes. Consider committing before deploying."
            echo "   Run: git status"
        else
            echo -e "${GREEN}✅${NC} Git repository is clean"
        fi
        
        # Check current branch
        BRANCH=$(git branch --show-current)
        echo -e "${BLUE}ℹ️${NC}  Current branch: $BRANCH"
        echo -e "${BLUE}ℹ️${NC}  Make sure this branch is connected to Render"
    else
        warn "Not a git repository - Render requires GitHub connection"
    fi
else
    warn "Git not found - cannot check repository status"
fi

echo ""
echo "5️⃣  Checking Frontend Build"
echo "---------------------------"

if [ -d "frontend/node_modules" ]; then
    echo -e "${GREEN}✅${NC} Frontend dependencies installed"
else
    warn "Frontend node_modules not found (will be installed during Render build)"
fi

# Check if frontend can build
if [ -f "frontend/package.json" ]; then
    if grep -q '"build"' frontend/package.json; then
        echo -e "${GREEN}✅${NC} Frontend has build script"
    else
        echo -e "${RED}❌${NC} Frontend missing build script"
        ((ERRORS++))
    fi
fi

echo ""
echo "6️⃣  Environment Variables Checklist"
echo "------------------------------------"

echo "Backend variables needed:"
echo "  - USE_LLM (true/false)"
echo "  - HOST (0.0.0.0)"
echo "  - PORT (8000, set automatically by Render)"
echo "  - RELOAD (false)"
echo "  - ALLOWED_HOSTS (your-backend.onrender.com)"
echo "  - CORS_ORIGINS (set after frontend deploys)"
echo ""
echo "Frontend variables needed:"
echo "  - VITE_API_URL (https://your-backend.onrender.com/api)"
echo ""
echo -e "${BLUE}ℹ️${NC}  See RENDER_ENV_VARS.md for complete reference"

echo ""
echo "7️⃣  Render Disk Requirements"
echo "----------------------------"

echo "You'll need to create Render Disks:"
echo "  1. querygenie-data (mount: /app/data, size: 1GB)"
echo "  2. querygenie-models (mount: /app/models, size: 1GB, if using LLM)"
echo ""
echo -e "${BLUE}ℹ️${NC}  Free tier: 1GB per disk, ~\$0.25/GB/month after"

echo ""
echo "==========================================="
echo "📊 Summary"
echo "==========================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Ready to deploy.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review RENDER_DEPLOYMENT.md"
    echo "  2. Create backend service on Render"
    echo "  3. Create frontend service on Render"
    echo "  4. Configure environment variables"
    echo "  5. Upload FAISS index (if pre-built)"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  $WARNINGS warning(s) found, but no errors.${NC}"
    echo "You can proceed, but review warnings above."
    echo ""
    echo "Next steps:"
    echo "  1. Review warnings above"
    echo "  2. Follow RENDER_DEPLOYMENT.md"
    exit 0
else
    echo -e "${RED}❌ $ERRORS error(s) and $WARNINGS warning(s) found.${NC}"
    echo "Please fix errors before deploying."
    exit 1
fi

