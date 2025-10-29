#!/bin/bash
echo "🚀 Starting QueryGenie in Development Mode..."

# Start backend in background
echo "Starting backend..."
cd backend
export PYTHONPATH="../src:$PYTHONPATH"
export USE_LLM=true
export LLM_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# export LLM_MODEL="microsoft/phi-2"
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