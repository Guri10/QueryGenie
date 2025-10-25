#!/bin/bash
echo "🚀 Starting QueryGenie Backend..."
cd backend
export PYTHONPATH="../src:$PYTHONPATH"
python main.py
