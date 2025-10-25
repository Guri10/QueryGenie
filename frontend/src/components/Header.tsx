import React from 'react'
import { Bot, Trash2, Github } from 'lucide-react'

interface HeaderProps {
  onClearChat: () => void
}

export const Header: React.FC<HeaderProps> = ({ onClearChat }) => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-xl flex items-center justify-center">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">QueryGenie</h1>
              <p className="text-sm text-gray-600">AI Research Assistant</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <button
              onClick={onClearChat}
              className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              Clear Chat
            </button>
            
            <a
              href="https://github.com/your-username/querygenie"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <Github className="w-4 h-4" />
              GitHub
            </a>
          </div>
        </div>
      </div>
    </header>
  )
}
