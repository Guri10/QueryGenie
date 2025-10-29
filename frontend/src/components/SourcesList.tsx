import React, { useState } from 'react'
import { ExternalLink, ChevronDown, ChevronUp, BookOpen } from 'lucide-react'
import { Source } from '../types'

interface SourcesListProps {
  sources: Source[]
  messageId?: string
}

export const SourcesList: React.FC<SourcesListProps> = ({ sources, messageId = 'default' }) => {
  const [isExpanded, setIsExpanded] = useState(false)
  const [expandedSource, setExpandedSource] = useState<number | null>(null)

  return (
    <div className="border-t border-gray-200 pt-3">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-sm font-medium text-gray-700 hover:text-gray-900 transition-colors"
      >
        <BookOpen className="w-4 h-4" />
        <span>Sources ({sources.length})</span>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4" />
        ) : (
          <ChevronDown className="w-4 h-4" />
        )}
      </button>
      
      {isExpanded && (
        <div className="mt-3 space-y-3">
          {sources.map((source, index) => (
            <div key={`${messageId}-${source.paper_id}-${index}`} className="bg-gray-50 rounded-lg p-3">
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-medium text-primary-600 bg-primary-100 px-2 py-1 rounded">
                      [{source.number}]
                    </span>
                    <span className="text-xs text-gray-500">
                      {(source.similarity_score * 100).toFixed(1)}% match
                    </span>
                  </div>
                  
                  <h4 className="font-medium text-sm text-gray-900 mb-1 line-clamp-2">
                    {source.paper_title}
                  </h4>
                  
                  <p className="text-xs text-gray-600 mb-2">
                    {source.authors.join(', ')}
                  </p>
                  
                  {expandedSource === index ? (
                    <div className="text-xs text-gray-700 bg-white rounded p-2 border">
                      {source.text_preview}
                    </div>
                  ) : (
                    <p className="text-xs text-gray-600 line-clamp-2">
                      {source.text_preview.substring(0, 150)}...
                    </p>
                  )}
                </div>
                
                <div className="flex flex-col gap-1">
                  <button
                    onClick={() => setExpandedSource(expandedSource === index ? null : index)}
                    className="text-xs text-primary-600 hover:text-primary-700 transition-colors"
                  >
                    {expandedSource === index ? 'Show less' : 'Show more'}
                  </button>
                  
                  <a
                    href={source.paper_id}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-primary-600 hover:text-primary-700 transition-colors flex items-center gap-1"
                  >
                    <ExternalLink className="w-3 h-3" />
                    View paper
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
