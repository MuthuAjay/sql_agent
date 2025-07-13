// frontend/src/services/api.ts
import axios, { AxiosInstance } from 'axios';
import type { 
  Database, 
  Table, 
  TableSchema,
  SampleData,
  QueryResult, 
  QueryHistory, 
  VisualizationSuggestion, 
  Connection, 
  UserPreferences, 
  ChatMessage,
  ValidationResult
} from '../types';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000, // 30 second timeout
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        config.headers['X-Request-ID'] = crypto.randomUUID();
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }
        
        // Enhanced error handling
        const errorMessage = error.response?.data?.error?.detail || 
                           error.response?.data?.detail || 
                           error.message || 
                           'An unknown error occurred';
        
        return Promise.reject(new Error(errorMessage));
      }
    );
  }

  // Query Processing APIs
  query = {
    naturalLanguage: async (query: string, databaseId: string, context?: any): Promise<QueryResult> => {
      const response = await this.client.post('/api/v1/query/natural-language', {
        query,
        database_id: databaseId,
        context: context || {},
      });
      return response.data;
    },

    validate: async (query: string, databaseId: string): Promise<ValidationResult> => {
      const response = await this.client.post('/api/v1/query/validate', {
        query,
        database_id: databaseId,
      });
      return response.data;
    },

    getHistory: async (
      page = 1, 
      limit = 20, 
      databaseId?: string
    ): Promise<{ 
      queries: QueryHistory[]; 
      total: number; 
      page: number; 
      limit: number; 
      pages: number; 
    }> => {
      const response = await this.client.get('/api/v1/query/history', {
        params: { page, limit, database_id: databaseId },
      });
      return response.data;
    },

    getById: async (queryId: string): Promise<QueryResult> => {
      const response = await this.client.get(`/api/v1/query/${queryId}`);
      return response.data;
    },
  };

  // SQL Management APIs
  sql = {
    execute: async (sql: string, databaseId: string, dryRun = false): Promise<QueryResult> => {
      const response = await this.client.post('/api/v1/sql/execute', {
        sql,
        database_id: databaseId,
        dry_run: dryRun,
      });
      return response.data;
    },

    explain: async (sql: string, databaseId: string): Promise<any> => {
      const response = await this.client.post('/api/v1/sql/explain', {
        sql,
        database_id: databaseId,
      });
      return response.data;
    },

    optimize: async (sql: string, databaseId: string): Promise<{ suggestions: string[] }> => {
      const response = await this.client.post('/api/v1/sql/optimize', {
        sql,
        database_id: databaseId,
      });
      return response.data;
    },

    format: async (sql: string): Promise<{ formatted: string }> => {
      const response = await this.client.post('/api/v1/sql/format', { sql });
      return response.data;
    },
  };

  // Schema Management APIs
  schema = {
    getDatabases: async (): Promise<Database[]> => {
      const response = await this.client.get('/api/v1/schema/databases');
      return response.data;
    },

    getTables: async (databaseId: string): Promise<{ tables: Table[] }> => {
      const response = await this.client.get(`/api/v1/schema/databases/${databaseId}/tables`);
      return response.data;
    },

    getTable: async (databaseId: string, tableName: string): Promise<Table> => {
      const response = await this.client.get(`/api/v1/schema/databases/${databaseId}/tables/${tableName}`);
      return response.data;
    },

    getTableSchema: async (databaseId: string, tableName: string): Promise<TableSchema> => {
      const response = await this.client.get(`/api/v1/schema/databases/${databaseId}/tables/${tableName}/schema`);
      return response.data;
    },

    getSampleData: async (databaseId: string, tableName: string, limit = 5): Promise<SampleData> => {
      const response = await this.client.get(`/api/v1/schema/databases/${databaseId}/tables/${tableName}/sample`, {
        params: { limit }
      });
      return response.data;
    },

    generateDescription: async (
      databaseId: string, 
      tableName: string, 
      regenerate = false
    ): Promise<{ description: string; generatedAt: string; cached: boolean }> => {
      const response = await this.client.post(`/api/v1/schema/databases/${databaseId}/tables/${tableName}/description`, {
        regenerate
      });
      return response.data;
    },

    searchSchema: async (databaseId: string, query: string, types = 'table,column'): Promise<any[]> => {
      const response = await this.client.get(`/api/v1/schema/databases/${databaseId}/search`, {
        params: { q: query, type: types },
      });
      return response.data;
    },

    refreshSchema: async (databaseId: string): Promise<void> => {
      await this.client.post(`/api/v1/schema/databases/${databaseId}/refresh`);
    },
  };

  // Analysis APIs
  analysis = {
    summarize: async (sql: string, databaseId: string, analysisType = 'statistical'): Promise<any> => {
      const response = await this.client.post('/api/v1/analysis/summarize', {
        sql,
        database_id: databaseId,
        analysis_type: analysisType,
      });
      return response.data;
    },

    profile: async (table: string, columns: string[], databaseId: string): Promise<any> => {
      const response = await this.client.post('/api/v1/analysis/profile', {
        table,
        columns,
        database_id: databaseId,
      });
      return response.data;
    },
  };

  // Visualization APIs
  visualization = {
    suggest: async (sql: string, databaseId: string): Promise<{ suggestions: VisualizationSuggestion[] }> => {
      const response = await this.client.post('/api/v1/visualization/suggest', {
        sql,
        database_id: databaseId,
      });
      return response.data;
    },

    generate: async (data: any[], chartType: string, config: any): Promise<any> => {
      const response = await this.client.post('/api/v1/visualization/generate', {
        data,
        chart_type: chartType,
        config,
      });
      return response.data;
    },
  };

  // Connection Management APIs
  connection = {
    getConnections: async (): Promise<Connection[]> => {
      const response = await this.client.get('/api/v1/connections');
      return response.data;
    },

    createConnection: async (connection: Omit<Connection, 'id' | 'status'>): Promise<Connection> => {
      const response = await this.client.post('/api/v1/connections', connection);
      return response.data;
    },

    testConnection: async (connectionId: string): Promise<{ success: boolean; message?: string }> => {
      const response = await this.client.post(`/api/v1/connections/${connectionId}/test`);
      return response.data;
    },

    updateConnection: async (connectionId: string, updates: Partial<Connection>): Promise<Connection> => {
      const response = await this.client.put(`/api/v1/connections/${connectionId}`, updates);
      return response.data;
    },

    deleteConnection: async (connectionId: string): Promise<void> => {
      await this.client.delete(`/api/v1/connections/${connectionId}`);
    },
  };

  // AI Assistant APIs
  assistant = {
    chat: async (message: string, context: any): Promise<ChatMessage> => {
      const response = await this.client.post('/api/v1/assistant/chat', {
        message,
        context,
      });
      return response.data;
    },

    suggestQuestions: async (table: string, databaseId: string): Promise<string[]> => {
      const response = await this.client.post('/api/v1/assistant/suggest-questions', {
        table,
        database_id: databaseId,
      });
      return response.data;
    },

    explainQuery: async (sql: string, databaseId: string): Promise<string> => {
      const response = await this.client.post('/api/v1/assistant/explain-query', {
        sql,
        database_id: databaseId,
      });
      return response.data;
    },
  };

  // User APIs
  user = {
    getProfile: async (): Promise<any> => {
      const response = await this.client.get('/api/v1/user/profile');
      return response.data;
    },

    updateProfile: async (profile: any): Promise<any> => {
      const response = await this.client.put('/api/v1/user/profile', profile);
      return response.data;
    },

    updatePreferences: async (preferences: UserPreferences): Promise<void> => {
      await this.client.put('/api/v1/user/preferences', preferences);
    },

    getPreferences: async (): Promise<UserPreferences> => {
      const response = await this.client.get('/api/v1/user/preferences');
      return response.data;
    },

    getFavorites: async (): Promise<any[]> => {
      const response = await this.client.get('/api/v1/user/favorites');
      return response.data;
    },

    addFavorite: async (item: any): Promise<void> => {
      await this.client.post('/api/v1/user/favorites', item);
    },

    removeFavorite: async (itemId: string): Promise<void> => {
      await this.client.delete(`/api/v1/user/favorites/${itemId}`);
    },
  };

  // Health check
  health = {
    check: async (): Promise<any> => {
      const response = await this.client.get('/health');
      return response.data;
    },
  };

  // Utility methods
  utils = {
    uploadFile: async (file: File, endpoint: string): Promise<any> => {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await this.client.post(endpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      return response.data;
    },

    downloadFile: async (endpoint: string, filename: string): Promise<void> => {
      const response = await this.client.get(endpoint, {
        responseType: 'blob',
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    },
  };
}

// Create singleton instance
const apiClient = new ApiClient();

// Export individual APIs for backward compatibility
export const queryAPI = apiClient.query;
export const sqlAPI = apiClient.sql;
export const schemaAPI = apiClient.schema;
export const analysisAPI = apiClient.analysis;
export const visualizationAPI = apiClient.visualization;
export const connectionAPI = apiClient.connection;
export const assistantAPI = apiClient.assistant;
export const userAPI = apiClient.user;

// Export the main client
export default apiClient;