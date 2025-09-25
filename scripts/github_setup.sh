#!/bin/bash

# QueryGenie GitHub Setup Script
# This script helps you set up the project for GitHub

echo "🚀 QueryGenie GitHub Setup"
echo "=========================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files
echo "📝 Adding files to git..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit."
else
    echo "💾 Committing changes..."
    git commit -m "Initial commit: QueryGenie RAG Chatbot

- Complete RAG system with arXiv integration
- FastAPI backend with web interface
- FAISS-based similarity search
- Docker support for easy deployment
- Zero-cost operation with local models
- 998 research papers indexed
- Sub-second query response times"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Name it 'QueryGenie'"
echo "   - Make it public or private"
echo "   - Don't initialize with README (we already have one)"
echo ""
echo "2. Add the remote origin:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/QueryGenie.git"
echo ""
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "4. Update README.md:"
echo "   - Replace 'YOUR_USERNAME' with your actual GitHub username"
echo "   - Update any other personal information"
echo ""
echo "✅ Setup complete! Your project is ready for GitHub."
