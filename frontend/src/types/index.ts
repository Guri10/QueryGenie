export interface ChatMessage {
  id: string
  type: 'user' | 'assistant'
  content: string
  sources?: Source[]
  metadata?: {
    retrievalTime: number
    generationTime: number
    totalTime: number
    modelUsed: string
  }
  timestamp: Date
}

export interface Source {
  number: number
  paper_title: string
  authors: string[]
  paper_id: string
  similarity_score: number
  text_preview: string
}

export interface QueryResponse {
  answer: string
  sources: Source[]
  retrieval_time: number
  generation_time: number
  total_time: number
  model_used: string
  timestamp: string
}

export interface QueryRequest {
  question: string
  k?: number
  max_context_length?: number
  max_answer_length?: number
}

export interface StatsResponse {
  status: string
  pipeline_stats: {
    faiss_stats: {
      total_papers: number
      faiss_index_count: number
      metadata_chunk_count: number
    }
  }
  timestamp: string
}
