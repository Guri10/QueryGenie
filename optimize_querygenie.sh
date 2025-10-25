#!/bin/bash

# QueryGenie FAISS Index Optimization Script
# Quick and easy way to optimize your FAISS index for better performance

set -e

echo "=================================================="
echo "   QueryGenie FAISS Index Optimization"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not activated${NC}"
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if index exists
if [ ! -f "data/faiss_index.faiss" ]; then
    echo -e "${RED}❌ Error: FAISS index not found at data/faiss_index.faiss${NC}"
    echo "Please run preprocessing first:"
    echo "  python src/preprocessing.py"
    exit 1
fi

echo "Current index found ✅"
echo ""

# Get current stats
echo "Analyzing current index..."
python -c "
import faiss
index = faiss.read_index('data/faiss_index.faiss')
print(f'  Vectors: {index.ntotal}')
print(f'  Dimensions: {index.d}')
print(f'  Type: Flat (exact search)')
" || {
    echo -e "${RED}❌ Error reading index${NC}"
    exit 1
}
echo ""

# Ask user for index type
echo "Choose optimization level:"
echo "  1) IVF - Balanced (2-3x faster, 97% accuracy) [RECOMMENDED]"
echo "  2) HNSW - Maximum Speed (4-5x faster, 98% accuracy)"
echo "  3) Skip optimization"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        INDEX_TYPE="ivf"
        echo -e "${GREEN}✅ Using IVF index${NC}"
        ;;
    2)
        INDEX_TYPE="hnsw"
        echo -e "${GREEN}✅ Using HNSW index${NC}"
        ;;
    3)
        echo "Skipping optimization"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "Starting optimization..."
echo ""

# Run optimization
python scripts/optimize_index.py --type $INDEX_TYPE

echo ""
echo "=================================================="
echo -e "${GREEN}✅ Optimization Complete!${NC}"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Review the benchmark results above"
echo "2. Test the new index:"
echo "   python -c \"from src.faiss_manager import FAISSManager; fm = FAISSManager(); print('✅ Index loaded:', fm.is_ready())\""
echo ""
echo "3. If satisfied, replace the original:"
echo "   mv data/faiss_index_${INDEX_TYPE}.faiss data/faiss_index.faiss"
echo ""
echo "4. Restart your API server"
echo ""
echo -e "${YELLOW}Note: Backup created at data/faiss_index_backup.faiss${NC}"
echo "=================================================="

