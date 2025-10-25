import React, { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, BookOpen, Clock, Database } from 'lucide-react'
import { QueryService } from './services/queryService'
import { ChatMessage, Source } from './types'
import { MessageBubble } from './components/MessageBubble'
import { SourcesList } from './components/SourcesList'
import { Header } from './components/Header'
import { StatsCard } from './components/StatsCard'

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [stats, setStats] = useState<any>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const queryService = new QueryService()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Load initial stats
    queryService.getStats().then(setStats).catch(console.error)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || isLoading) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      const response = await queryService.askQuestion(inputValue)
      
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.answer,
        sources: response.sources,
        metadata: {
          retrievalTime: response.retrieval_time,
          generationTime: response.generation_time,
          totalTime: response.total_time,
          modelUsed: response.model_used
        },
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error asking question:', error)
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error while processing your question. Please try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const clearChat = () => {
    setMessages([])
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-secondary-50">
      <Header onClearChat={clearChat} />
      
      <main className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <StatsCard
              title="Total Papers"
              value={stats.pipeline_stats?.faiss_stats?.total_papers || 0}
              icon={<Database className="w-5 h-5" />}
              color="blue"
            />
            <StatsCard
              title="Index Vectors"
              value={stats.pipeline_stats?.faiss_stats?.faiss_index_count || 0}
              icon={<BookOpen className="w-5 h-5" />}
              color="green"
            />
            <StatsCard
              title="System Status"
              value={stats.status}
              icon={<Clock className="w-5 h-5" />}
              color="purple"
            />
          </div>
        )}

        {/* Chat Container */}
        <div className="card">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-primary-100 rounded-full flex items-center justify-center">
              <Bot className="w-5 h-5 text-primary-600" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">QueryGenie Assistant</h2>
              <p className="text-sm text-gray-500">Ask me anything about research papers!</p>
            </div>
          </div>

          {/* Messages */}
          <div className="space-y-4 mb-6 max-h-96 overflow-y-auto">
            {messages.length === 0 && (
              <div className="text-center py-12 text-gray-500">
                <Bot className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p className="text-lg font-medium">Welcome to QueryGenie!</p>
                <p className="text-sm">Start by asking a question about research papers.</p>
              </div>
            )}
            
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            
            {isLoading && (
              <div className="message-bubble assistant-message">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Thinking...</span>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input Form */}
          <form onSubmit={handleSubmit} className="flex gap-3">
            <div className="flex-1 relative">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Ask about research papers..."
                className="input-field pr-12"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || isLoading}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 text-primary-600 hover:text-primary-700 disabled:opacity-50"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </form>
        </div>
      </main>
    </div>
  )
}

export default App
