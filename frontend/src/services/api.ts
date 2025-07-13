// frontend/src/services/api.ts
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import type { 
  Database, 
  Table, 
  QueryResult, 
  QueryHistory, 
  VisualizationSuggestion, 
  Connection, 
  UserPreferences, 
  ChatMessage 
} from '../types';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

interface RequestOptions {
  timeout?: number;
  signal?: AbortSignal;
}

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000, // 30 seconds default timeout
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

  private async request<T>(config: AxiosRequestConfig & RequestOptions): Promise<T> {
    try {
      const response = await this.client(config);
      return response.data;
    } catch (error) {
      throw error;
    }
  }

  // Query Processing APIs
  query = {
    naturalLanguage: async (query: string, databaseId: string, context?: any): Promise<QueryResult> => {
      return this.request<QueryResult>({
        method: 'POST',
        url: '/api/v1/query',
        data: {
          query,
          database_name: databaseId,
          context: context || {},
          include_analysis: true,
          include_visualization: false
        }
      });
    },

    processSimple: async (request: {
      query: string;
      database_name?: string;
      max_results?: number;
      include_analysis?: boolean;
      include_visualization?: boolean;
    }): Promise<any> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/query/simple',
        data: request
      });
    },

    validate: async (query: string, databaseId: string): Promise<{ valid: boolean; suggestions?: string[] }> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/query/validate',
        data: {
          query,
          database_id: databaseId,
        },
      });
    },

    getHistory: async (page = 1, limit = 20, databaseId?: string): Promise<{ queries: QueryHistory[]; total: number }> => {
      return this.request({
        method: 'GET',
        url: '/api/v1/query/history',
        params: { page, limit, database_id: databaseId },
      });
    },

    getById: async (queryId: string): Promise<QueryResult> => {
      return this.request({
        method: 'GET',
        url: `/api/v1/query/${queryId}`,
      });
    },

    clearHistory: async (): Promise<{ message: string }> => {
      return this.request({
        method: 'DELETE',
        url: '/api/v1/query/history',
        params: { confirm: true }
      });
    }
  };

  // SQL Management APIs
  sql = {
    generate: async (request: {
      query: string;
      database_name?: string;
      include_explanation?: boolean;
      include_validation?: boolean;
      optimize?: boolean;
      context?: any;
    }): Promise<{
      query: string;
      generated_sql: string;
      explanation?: string;
      processing_time: number;
      has_errors: boolean;
      errors: string[];
    }> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/sql/generate',
        data: request
      });
    },

    execute: async (sql: string, databaseId: string, dryRun = false): Promise<QueryResult> => {
      return this.request<QueryResult>({
        method: 'POST',
        url: '/api/v1/sql/execute',
        data: {
          sql,
          database_name: databaseId,
          dry_run: dryRun,
        },
      });
    },

    validate: async (sql: string, databaseId: string): Promise<{
      sql: string;
      is_valid: boolean;
      errors: string[];
      warnings: string[];
      suggestions: string[];
    }> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/sql/validate',
        data: {
          sql,
          database_name: databaseId,
        },
      });
    },

    explain: async (sql: string, databaseId: string): Promise<any> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/sql/explain',
        data: {
          sql,
          database_id: databaseId,
        },
      });
    },

    optimize: async (sql: string, databaseId: string): Promise<{ suggestions: string[] }> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/sql/optimize',
        data: {
          sql,
          database_id: databaseId,
        },
      });
    },

    format: async (sql: string): Promise<{ formatted: string }> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/sql/format',
        data: { sql },
      });
    },

    getTemplates: async (): Promise<{
      templates: Record<string, {
        name: string;
        template: string;
        description: string;
      }>;
      total: number;
    }> => {
      return this.request({
        method: 'GET',
        url: '/api/v1/sql/templates'
      });
    }
  };

  // Schema Management APIs
  schema = {
    getDatabases: async (): Promise<Database[]> => {
      return this.request<Database[]>({
        method: 'GET',
        url: '/api/v1/schema/databases',
      });
    },

    getTables: async (databaseId: string): Promise<{ tables: Table[] }> => {
      return this.request({
        method: 'GET',
        url: `/api/v1/schema/databases/${databaseId}/tables`,
      });
    },

    getTable: async (databaseId: string, tableName: string): Promise<Table> => {
      return this.request<Table>({
        method: 'GET',
        url: `/api/v1/schema/databases/${databaseId}/tables/${tableName}`,
      });
    },

    getTableSchema: async (databaseId: string, tableName: string): Promise<any> => {
      return this.request({
        method: 'GET',
        url: `/api/v1/schema/databases/${databaseId}/tables/${tableName}/schema`,
      });
    },

    getSampleData: async (databaseId: string, tableName: string, limit = 5): Promise<{
      columns: string[];
      data: any[][];
      total_rows: number;
      sample_size: number;
    }> => {
      return this.request({
        method: 'GET',
        url: `/api/v1/schema/databases/${databaseId}/tables/${tableName}/sample`,
        params: { limit }
      });
    },

    generateDescription: async (
      databaseId: string, 
      tableName: string, 
      regenerate = false
    ): Promise<{ description: string; generatedAt: string; cached: boolean }> => {
      return this.request({
        method: 'POST',
        url: `/api/v1/schema/databases/${databaseId}/tables/${tableName}/description`,
        data: { regenerate }
      });
    },

    searchSchema: async (databaseId: string, query: string, types = 'table,column'): Promise<any[]> => {
      return this.request<any[]>({
        method: 'GET',
        url: `/api/v1/schema/databases/${databaseId}/search`,
        params: { q: query, type: types },
      });
    },

    refreshSchema: async (databaseId: string): Promise<{ message: string }> => {
      return this.request({
        method: 'POST',
        url: `/api/v1/schema/databases/${databaseId}/refresh`,
      });
    }
  };

  // Analysis APIs
  analysis = {
    summarize: async (sql: string, databaseId: string, analysisType = 'comprehensive'): Promise<any> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/analysis/analyze',
        data: {
          sql,
          database_id: databaseId,
          analysis_type: analysisType,
        },
      });
    },

    analyzeSQLResults: async (data: any[], queryContext?: string, analysisType = 'comprehensive'): Promise<any> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/analysis/analyze/sql',
        data: {
          data,
          query_context: queryContext,
          analysis_type: analysisType,
        },
      });
    },

    profile: async (table: string, columns: string[], databaseId: string): Promise<any> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/analysis/profile',
        data: {
          table,
          columns,
          database_id: databaseId,
        },
      });
    },

    getTypes: async (): Promise<{
      analysis_types: Record<string, {
        name: string;
        description: string;
        includes: string[];
      }>;
      total: number;
    }> => {
      return this.request({
        method: 'GET',
        url: '/api/v1/analysis/types'
      });
    }
  };

  // Visualization APIs
  visualization = {
    suggest: async (sql: string, databaseId: string): Promise<{ suggestions: VisualizationSuggestion[] }> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/visualization/suggest',
        data: {
          sql,
          database_id: databaseId,
        },
      });
    },

    create: async (request: {
      data: any[];
      chart_type?: string;
      title?: string;
      x_axis?: string;
      y_axis?: string;
      color_by?: string;
      size_by?: string;
      filters?: any;
      aggregation?: string;
      theme?: string;
      interactive?: boolean;
    }): Promise<any> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/visualization/create',
        data: request
      });
    },

    generate: async (data: any[], chartType: string, config: any): Promise<any> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/visualization/generate',
        data: {
          data,
          chart_type: chartType,
          config,
        },
      });
    },

    getTypes: async (): Promise<{
      chart_types: Record<string, {
        name: string;
        description: string;
        best_for: string[];
        data_requirements: string[];
      }>;
      total: number;
    }> => {
      return this.request({
        method: 'GET',
        url: '/api/v1/visualization/types'
      });
    },

    export: async (request: {
      data: any[];
      chart_type?: string;
      title?: string;
      x_axis?: string;
      y_axis?: string;
    }, format: string = 'json'): Promise<{
      format: string;
      data?: any;
      content?: string;
    }> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/visualization/export',
        data: request,
        params: { format }
      });
    }
  };

  // Connection Management APIs
  connection = {
    getConnections: async (): Promise<Connection[]> => {
      return this.request<Connection[]>({
        method: 'GET',
        url: '/api/v1/connections',
      });
    },

    createConnection: async (connection: Omit<Connection, 'id' | 'status'>): Promise<Connection> => {
      return this.request<Connection>({
        method: 'POST',
        url: '/api/v1/connections',
        data: connection,
      });
    },

    updateConnection: async (connectionId: string, updates: Partial<Connection>): Promise<Connection> => {
      return this.request<Connection>({
        method: 'PUT',
        url: `/api/v1/connections/${connectionId}`,
        data: updates,
      });
    },

    deleteConnection: async (connectionId: string): Promise<void> => {
      return this.request({
        method: 'DELETE',
        url: `/api/v1/connections/${connectionId}`,
      });
    },

    testConnection: async (connectionId: string): Promise<{ success: boolean; message?: string }> => {
      return this.request({
        method: 'POST',
        url: `/api/v1/connections/${connectionId}/test`,
      });
    }
  };

  // AI Assistant APIs
  assistant = {
    chat: async (message: string, context: any): Promise<ChatMessage> => {
      return this.request<ChatMessage>({
        method: 'POST',
        url: '/api/v1/assistant/chat',
        data: {
          message,
          context,
        },
      });
    },

    suggestQuestions: async (table: string, databaseId: string): Promise<string[]> => {
      return this.request<string[]>({
        method: 'POST',
        url: '/api/v1/assistant/suggest-questions',
        data: {
          table,
          database_id: databaseId,
        },
      });
    },

    explainQuery: async (sql: string, databaseId: string): Promise<string> => {
      return this.request<string>({
        method: 'POST',
        url: '/api/v1/assistant/explain-query',
        data: {
          sql,
          database_id: databaseId,
        },
      });
    }
  };

  // User APIs
  user = {
    getProfile: async (): Promise<any> => {
      return this.request({
        method: 'GET',
        url: '/api/v1/user/profile',
      });
    },

    updateProfile: async (profile: any): Promise<any> => {
      return this.request({
        method: 'PUT',
        url: '/api/v1/user/profile',
        data: profile,
      });
    },

    updatePreferences: async (preferences: UserPreferences): Promise<void> => {
      return this.request({
        method: 'PUT',
        url: '/api/v1/user/preferences',
        data: preferences,
      });
    },

    getPreferences: async (): Promise<UserPreferences> => {
      return this.request<UserPreferences>({
        method: 'GET',
        url: '/api/v1/user/preferences',
      });
    },

    getFavorites: async (): Promise<any[]> => {
      return this.request<any[]>({
        method: 'GET',
        url: '/api/v1/user/favorites',
      });
    },

    addFavorite: async (item: any): Promise<void> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/user/favorites',
        data: item,
      });
    },

    removeFavorite: async (itemId: string): Promise<void> => {
      return this.request({
        method: 'DELETE',
        url: `/api/v1/user/favorites/${itemId}`,
      });
    }
  };

  // Health and System APIs
  health = {
    check: async (): Promise<{
      status: string;
      timestamp: number;
      version: string;
      services: Record<string, string>;
      metrics?: any;
      uptime?: number;
    }> => {
      return this.request({
        method: 'GET',
        url: '/health',
      });
    }
  };

  system = {
    getInfo: async (): Promise<{
      name: string;
      version: string;
      description: string;
      features: string[];
      endpoints: Record<string, string>;
      documentation: string;
      health: string;
    }> => {
      return this.request({
        method: 'GET',
        url: '/api/v1/info',
      });
    },

    getStatus: async (): Promise<{
      api_status: string;
      uptime: number;
      database_connected: boolean;
      orchestrator_ready: boolean;
      version: string;
      environment: string;
    }> => {
      return this.request({
        method: 'GET',
        url: '/api/v1/status',
      });
    }
  };

  // Utility methods
  utils = {
    uploadFile: async (file: File, endpoint: string): Promise<any> => {
      const formData = new FormData();
      formData.append('file', file);
      
      return this.request({
        method: 'POST',
        url: endpoint,
        data: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
    },

    downloadFile: async (endpoint: string, filename: string): Promise<void> => {
      try {
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
      } catch (error) {
        throw error;
      }
    },

    exportData: async (data: any[], format: 'csv' | 'json' | 'xlsx', filename: string): Promise<void> => {
      return this.request({
        method: 'POST',
        url: '/api/v1/export',
        data: {
          data,
          format,
          filename,
        },
        responseType: 'blob',
      }).then(() => {
        // File download handled by server
      });
    }
  };

  // WebSocket connection helper
  createWebSocket = (path: string, onMessage?: (data: any) => void, onError?: (error: Event) => void): WebSocket => {
    const wsUrl = API_BASE_URL.replace(/^http/, 'ws') + path;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage?.(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      onError?.(error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    return ws;
  };

  // Request cancellation
  createCancelToken = () => {
    const controller = new AbortController();
    return {
      token: controller.signal,
      cancel: (reason?: string) => controller.abort(reason)
    };
  };
}

// Create singleton instance
const apiClient = new ApiClient();

// Export the main client and individual API groups
export default apiClient;

// Named exports for convenience
export const queryAPI = apiClient.query;
export const sqlAPI = apiClient.sql;
export const schemaAPI = apiClient.schema;
export const analysisAPI = apiClient.analysis;
export const visualizationAPI = apiClient.visualization;
export const connectionAPI = apiClient.connection;
export const assistantAPI = apiClient.assistant;
export const userAPI = apiClient.user;
export const healthAPI = apiClient.health;
export const systemAPI = apiClient.system;
export const utilsAPI = apiClient.utils;

// Legacy exports for backward compatibility
export const tableApi = {
  getTables: (databaseId: string) => schemaAPI.getTables(databaseId),
  getTableSchema: (databaseId: string, tableName: string) => schemaAPI.getTableSchema(databaseId, tableName),
  getSampleData: (databaseId: string, tableName: string, limit = 5) => schemaAPI.getSampleData(databaseId, tableName, limit),
  generateDescription: (databaseId: string, tableName: string, regenerate = false) => 
    schemaAPI.generateDescription(databaseId, tableName, regenerate)
};

// Export types for consumers
export type { 
  Database, 
  Table, 
  QueryResult, 
  QueryHistory, 
  VisualizationSuggestion, 
  Connection, 
  UserPreferences, 
  ChatMessage 
} from '../types';