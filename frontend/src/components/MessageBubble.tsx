import React from 'react'
import { User, Bot, Clock, Zap } from 'lucide-react'
import { ChatMessage } from '../types'
import { SourcesList } from './SourcesList'

interface MessageBubbleProps {
  message: ChatMessage
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.type === 'user'
  const Icon = isUser ? User : Bot

  return (
    <div className={`message-bubble ${isUser ? 'user-message' : 'assistant-message'}`}>
      <div className="flex items-start gap-3">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser ? 'bg-primary-200' : 'bg-gray-200'
        }`}>
          <Icon className={`w-4 h-4 ${isUser ? 'text-primary-600' : 'text-gray-600'}`} />
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium text-sm">
              {isUser ? 'You' : 'QueryGenie'}
            </span>
            <span className="text-xs text-gray-500">
              {message.timestamp.toLocaleTimeString()}
            </span>
          </div>
          
          <div className="prose prose-sm max-w-none">
            <p className="whitespace-pre-wrap">{message.content}</p>
          </div>
          
          {message.metadata && (
            <div className="mt-3 flex items-center gap-4 text-xs text-gray-500">
              <div className="flex items-center gap-1">
                <Zap className="w-3 h-3" />
                <span>{message.metadata.totalTime.toFixed(2)}s</span>
              </div>
              <div className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                <span>{message.metadata.modelUsed}</span>
              </div>
            </div>
          )}
          
          {message.sources && message.sources.length > 0 && (
            <div className="mt-4">
              <SourcesList sources={message.sources} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
