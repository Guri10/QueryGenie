#!/bin/bash

# QueryGenie Frontend-Backend Setup Script
echo "🚀 Setting up QueryGenie Frontend & Backend..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+ first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create environment file
echo "⚙️ Creating environment configuration..."
if [ ! -f .env ]; then
    cp env.example .env
    echo "✅ Created .env file from template"
else
    echo "✅ .env file already exists"
fi

# Check if data exists
if [ ! -d "data" ] || [ ! -f "data/faiss_index.faiss" ]; then
    echo "⚠️  FAISS index not found. Please run preprocessing first:"
    echo "   python src/preprocessing.py"
    echo ""
    echo "   Or download some papers first:"
    echo "   python src/arxiv_downloader.py"
    exit 1
fi

echo "✅ Data directory and FAISS index found"

# Create startup scripts
echo "📝 Creating startup scripts..."

# Backend startup script
cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting QueryGenie Backend..."
cd backend
export PYTHONPATH="../src:$PYTHONPATH"
python main.py
EOF
chmod +x start_backend.sh

# Frontend startup script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting QueryGenie Frontend..."
cd frontend
npm run dev
EOF
chmod +x start_frontend.sh

# Development startup script
cat > start_dev.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting QueryGenie in Development Mode..."

# Start backend in background
echo "Starting backend..."
cd backend
export PYTHONPATH="../src:$PYTHONPATH"
python main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "Starting frontend..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo "✅ QueryGenie is running!"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user to stop
wait
EOF
chmod +x start_dev.sh

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick Start Options:"
echo ""
echo "1. Start everything (recommended):"
echo "   ./start_dev.sh"
echo ""
echo "2. Start backend only:"
echo "   ./start_backend.sh"
echo ""
echo "3. Start frontend only:"
echo "   ./start_frontend.sh"
echo ""
echo "4. Production deployment:"
echo "   docker-compose -f docker-compose.prod.yml up -d"
echo ""
echo "🌐 Access URLs:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Happy coding! 🚀"
