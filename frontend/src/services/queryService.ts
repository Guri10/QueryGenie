import axios from 'axios'
import { QueryRequest, QueryResponse, StatsResponse } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

class QueryService {
  private api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 120000, // 2 minutes timeout for LLM queries
    headers: {
      'Content-Type': 'application/json',
    },
  })

  async askQuestion(question: string, options?: Partial<QueryRequest>): Promise<QueryResponse> {
    try {
      const response = await this.api.post<QueryResponse>('/ask', {
        question,
        k: 5,
        max_context_length: 5000,
        max_answer_length: 300,
        ...options,
      })
      return response.data
    } catch (error) {
      console.error('Error asking question:', error)
      throw new Error('Failed to get answer from QueryGenie')
    }
  }

  async getStats(): Promise<StatsResponse> {
    try {
      const response = await this.api.get<StatsResponse>('/metrics')
      return response.data
    } catch (error) {
      console.error('Error fetching stats:', error)
      throw new Error('Failed to fetch system statistics')
    }
  }

  async getHealth(): Promise<{ status: string }> {
    try {
      const response = await this.api.get<{ status: string }>('/health')
      return response.data
    } catch (error) {
      console.error('Error checking health:', error)
      throw new Error('Failed to check system health')
    }
  }
}

export { QueryService }
