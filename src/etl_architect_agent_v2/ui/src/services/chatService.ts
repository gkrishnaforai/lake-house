import axios from 'axios';
import { getUserId } from './catalogService';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

export interface ChatMessage {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
}

export interface ChatResponse {
  message: string;
  data?: any;
  error?: string;
}

class ChatService {
  private async makeRequest(endpoint: string, data: any): Promise<ChatResponse> {
    try {
      const userId = getUserId();
      const response = await axios.post(`${API_BASE_URL}${endpoint}`, {
        ...data,
        user_id: userId
      });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.error || 'Failed to process chat request');
      }
      throw error;
    }
  }

  async sendMessage(message: string): Promise<ChatResponse> {
    return this.makeRequest('/api/chat', { message });
  }

  async getDataPreview(tableName: string): Promise<ChatResponse> {
    return this.makeRequest('/api/data/preview', { table_name: tableName });
  }

  async getDataQuality(tableName: string): Promise<ChatResponse> {
    return this.makeRequest('/api/data/quality', { table_name: tableName });
  }

  async getSchemaInfo(tableName: string): Promise<ChatResponse> {
    return this.makeRequest('/api/data/schema', { table_name: tableName });
  }

  async executeQuery(query: string): Promise<ChatResponse> {
    return this.makeRequest('/api/data/query', { query });
  }
}

export const chatService = new ChatService(); 