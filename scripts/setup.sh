#!/bin/bash

# QueryGenie Setup Script
# Automated setup for the QueryGenie RAG system

set -e  # Exit on any error

echo "🚀 Setting up QueryGenie RAG Chatbot..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.9+ is available
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.9+ is required but not found. Please install Python 3.9 or later."
        exit 1
    fi
}

# Create virtual environment
setup_venv() {
    print_status "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data
    mkdir -p logs
    mkdir -p web
    
    print_success "Directories created"
}

# Download and process papers
setup_data() {
    print_status "Setting up data (this may take a while)..."
    
    # Download papers
    print_status "Downloading papers from arXiv..."
    python src/arxiv_downloader.py
    
    # Process papers
    print_status "Processing papers and creating index..."
    python src/preprocessing.py
    
    print_success "Data setup completed"
}

# Setup cron job
setup_cron() {
    print_status "Setting up automated refresh..."
    
    chmod +x scripts/setup_cron.sh
    ./scripts/setup_cron.sh
    
    print_success "Cron job configured"
}

# Test the system
test_system() {
    print_status "Testing the system..."
    
    # Test FAISS manager
    python -c "
from src.faiss_manager import FAISSManager
manager = FAISSManager()
if manager.is_ready():
    print('✅ FAISS manager working')
else:
    print('❌ FAISS manager not ready')
    exit(1)
"
    
    # Test RAG pipeline
    python -c "
from src.faiss_manager import FAISSManager
from src.rag_pipeline import RAGPipeline
faiss_manager = FAISSManager()
rag = RAGPipeline(faiss_manager)
print('✅ RAG pipeline working')
"
    
    print_success "System test completed"
}

# Main setup function
main() {
    echo "🎯 QueryGenie RAG Chatbot Setup"
    echo "================================="
    
    # Check requirements
    check_python
    
    # Setup environment
    setup_venv
    install_dependencies
    create_directories
    
    # Setup data (optional, can be skipped)
    read -p "Do you want to download and process papers now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_data
    else
        print_warning "Skipping data setup. You can run this later with:"
        print_warning "  python src/arxiv_downloader.py"
        print_warning "  python src/preprocessing.py"
    fi
    
    # Setup automation
    read -p "Do you want to setup automated nightly refresh? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_cron
    fi
    
    # Test system if data was set up
    if [ -f "data/faiss_index.faiss" ]; then
        test_system
    fi
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Start the API server: python src/api.py"
    echo "3. Open http://localhost:8000/docs for API documentation"
    echo "4. Test with: curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json' -d '{\"question\": \"What is machine learning?\"}'"
    echo ""
    echo "For Docker deployment:"
    echo "  docker-compose up -d"
    echo ""
    echo "Happy querying! 🚀"
}

# Run main function
main "$@"
