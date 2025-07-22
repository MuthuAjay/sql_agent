import { QueryResponse, SchemaResponse, QueryOptions, PerformanceTestResult, HealthStatus, ErrorResponse } from '@/types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class APIError extends Error {
  constructor(public response: ErrorResponse) {
    super(response.error.user_message || response.error.detail);
    this.name = 'APIError';
  }
}

async function apiRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData: ErrorResponse = await response.json().catch(() => ({
        error: {
          type: 'network_error',
          status_code: response.status,
          detail: response.statusText,
          user_message: `Request failed with status ${response.status}`,
          suggestions: ['Please check your connection and try again'],
          request_id: '',
          timestamp: Date.now(),
        }
      }));
      throw new APIError(errorData);
    }

    return response.json();
  } catch (error) {
    if (error instanceof APIError) throw error;
    
    // Network or other errors
    throw new APIError({
      error: {
        type: 'network_error',
        status_code: 0,
        detail: error instanceof Error ? error.message : 'Unknown error',
        user_message: 'Unable to connect to the server. Please check your connection.',
        suggestions: ['Check your internet connection', 'Verify the server is running'],
        request_id: '',
        timestamp: Date.now(),
      }
    });
  }
}

export const api = {
  query: {
    process: async (query: string, options: QueryOptions = {}): Promise<QueryResponse> => {
      return apiRequest<QueryResponse>('/api/v1/query/process', {
        method: 'POST',
        body: JSON.stringify({
          query,
          session_id: options.sessionId || generateSessionId(),
          database_name: options.database || 'default',
          include_analysis: options.includeAnalysis ?? true,
          include_visualization: options.includeVisualization ?? true,
          max_results: options.maxResults || 100,
        }),
      });
    },
    
    getSuggestions: async (partialQuery: string): Promise<string[]> => {
      const response = await apiRequest<{ suggestions: string[] }>('/api/v1/query/suggestions', {
        method: 'POST',
        body: JSON.stringify({ partial_query: partialQuery }),
      });
      return response.suggestions;
    },
    
    validate: async (query: string): Promise<{ valid: boolean; errors: string[]; warnings: string[] }> => {
      return apiRequest('/api/v1/query/validate', {
        method: 'POST',
        body: JSON.stringify({ query }),
      });
    },
  },
  
  sql: {
    execute: async (sql: string, options: QueryOptions = {}): Promise<QueryResponse> => {
      return apiRequest<QueryResponse>('/api/v1/sql/execute', {
        method: 'POST',
        body: JSON.stringify({
          sql,
          database_name: options.database || 'default',
          max_results: options.maxResults || 100,
        }),
      });
    },
    
    getTemplates: async (): Promise<Array<{ id: string; name: string; description: string; sql: string; category: string }>> => {
      const response = await apiRequest<{ templates: any[] }>('/api/v1/sql/templates');
      return response.templates;
    },
  },
  
  schema: {
    get: async (database?: string): Promise<SchemaResponse> => {
      const params = database ? `?database=${database}` : '';
      return apiRequest<SchemaResponse>(`/api/v1/schema/${params}`);
    },
    
    getTable: async (tableName: string, database?: string): Promise<TableInfo> => {
      const params = database ? `?database=${database}` : '';
      return apiRequest(`/api/v1/schema/tables/${tableName}${params}`);
    },
    
    searchTables: async (query: string): Promise<TableInfo[]> => {
      const response = await apiRequest<{ tables: TableInfo[] }>('/api/v1/schema/search', {
        method: 'POST',
        body: JSON.stringify({ query }),
      });
      return response.tables;
    },
  },
  
  performance: {
    test: async (numQueries: number): Promise<PerformanceTestResult> => {
      return apiRequest<PerformanceTestResult>('/api/v1/performance/test', {
        method: 'POST',
        body: JSON.stringify({ num_queries: numQueries }),
      });
    },
    
    getMetrics: async (): Promise<{
      avg_response_time: number;
      total_queries: number;
      error_rate: number;
      uptime: number;
    }> => {
      return apiRequest('/api/v1/performance/metrics');
    },
  },
  
  health: {
    check: async (): Promise<HealthStatus> => {
      return apiRequest<HealthStatus>('/api/v1/health');
    },
  },
};

function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

export { APIError };