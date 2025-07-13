import axios from 'axios';
import type { Database, Table, QueryResult, QueryHistory, VisualizationSuggestion, Connection, UserPreferences, ChatMessage } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for auth
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  config.headers['X-Request-ID'] = crypto.randomUUID();
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Query Processing APIs
export const queryAPI = {
  naturalLanguage: async (query: string, databaseId: string): Promise<QueryResult> => {
    const response = await api.post('/api/v1/query/natural-language', {
      query,
      database_id: databaseId,
      context: {
        previous_queries: [],
        user_preferences: {},
      },
    });
    return response.data;
  },

  validate: async (query: string, databaseId: string): Promise<{ valid: boolean; suggestions?: string[] }> => {
    const response = await api.post('/api/v1/query/validate', {
      query,
      database_id: databaseId,
    });
    return response.data;
  },

  getHistory: async (page = 1, limit = 20, databaseId?: string): Promise<{ queries: QueryHistory[]; total: number }> => {
    const response = await api.get('/api/v1/query/history', {
      params: { page, limit, database_id: databaseId },
    });
    return response.data;
  },

  getById: async (queryId: string): Promise<QueryResult> => {
    const response = await api.get(`/api/v1/query/${queryId}`);
    return response.data;
  },
};

// SQL Management APIs
export const sqlAPI = {
  execute: async (sql: string, databaseId: string, dryRun = false): Promise<QueryResult> => {
    const response = await api.post('/api/v1/sql/execute', {
      sql,
      database_id: databaseId,
      dry_run: dryRun,
    });
    return response.data;
  },

  explain: async (sql: string, databaseId: string): Promise<any> => {
    const response = await api.post('/api/v1/sql/explain', {
      sql,
      database_id: databaseId,
    });
    return response.data;
  },

  optimize: async (sql: string, databaseId: string): Promise<{ suggestions: string[] }> => {
    const response = await api.post('/api/v1/sql/optimize', {
      sql,
      database_id: databaseId,
    });
    return response.data;
  },

  format: async (sql: string): Promise<{ formatted: string }> => {
    const response = await api.post('/api/v1/sql/format', { sql });
    return response.data;
  },
};

// Schema Management APIs
export const schemaAPI = {
  getDatabases: async (): Promise<Database[]> => {
    const response = await api.get('/api/v1/schema/databases');
    return response.data;
  },

  getTables: async (databaseId: string): Promise<Table[]> => {
    const response = await api.get(`/api/v1/schema/${databaseId}/tables`);
    return response.data;
  },

  getTable: async (databaseId: string, tableName: string): Promise<Table> => {
    const response = await api.get(`/api/v1/schema/${databaseId}/tables/${tableName}`);
    return response.data;
  },

  searchSchema: async (databaseId: string, query: string): Promise<any[]> => {
    const response = await api.get(`/api/v1/schema/${databaseId}/search`, {
      params: { q: query, type: 'table,column' },
    });
    return response.data;
  },

  refreshSchema: async (databaseId: string): Promise<void> => {
    await api.post(`/api/v1/schema/${databaseId}/refresh`);
  },
};

// Analysis APIs
export const analysisAPI = {
  summarize: async (sql: string, databaseId: string): Promise<any> => {
    const response = await api.post('/api/v1/analysis/summarize', {
      sql,
      database_id: databaseId,
      analysis_type: 'statistical',
    });
    return response.data;
  },

  profile: async (table: string, columns: string[], databaseId: string): Promise<any> => {
    const response = await api.post('/api/v1/analysis/profile', {
      table,
      columns,
      database_id: databaseId,
    });
    return response.data;
  },
};

// Visualization APIs
export const visualizationAPI = {
  suggest: async (sql: string, databaseId: string): Promise<{ suggestions: VisualizationSuggestion[] }> => {
    const response = await api.post('/api/v1/visualization/suggest', {
      sql,
      database_id: databaseId,
    });
    return response.data;
  },

  generate: async (data: any[], chartType: string, config: any): Promise<any> => {
    const response = await api.post('/api/v1/visualization/generate', {
      data,
      chart_type: chartType,
      config,
    });
    return response.data;
  },
};

// Connection Management APIs
export const connectionAPI = {
  getConnections: async (): Promise<Connection[]> => {
    const response = await api.get('/api/v1/connections');
    return response.data;
  },

  createConnection: async (connection: Omit<Connection, 'id' | 'status'>): Promise<Connection> => {
    const response = await api.post('/api/v1/connections', connection);
    return response.data;
  },

  testConnection: async (connectionId: string): Promise<{ success: boolean; message?: string }> => {
    const response = await api.post(`/api/v1/connections/${connectionId}/test`);
    return response.data;
  },
};

// AI Assistant APIs
export const assistantAPI = {
  chat: async (message: string, context: any): Promise<ChatMessage> => {
    const response = await api.post('/api/v1/assistant/chat', {
      message,
      context,
    });
    return response.data;
  },

  suggestQuestions: async (table: string, databaseId: string): Promise<string[]> => {
    const response = await api.post('/api/v1/assistant/suggest-questions', {
      table,
      database_id: databaseId,
    });
    return response.data;
  },
};

// User APIs
export const userAPI = {
  getProfile: async (): Promise<any> => {
    const response = await api.get('/api/v1/user/profile');
    return response.data;
  },

  updatePreferences: async (preferences: UserPreferences): Promise<void> => {
    await api.put('/api/v1/user/preferences', preferences);
  },

  getFavorites: async (): Promise<any[]> => {
    const response = await api.get('/api/v1/user/favorites');
    return response.data;
  },
};

export default api;

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const tableApi = {
  // Get all tables for a database
  getTables: async (databaseId: string): Promise<{ tables: Table[] }> => {
    const response = await fetch(`${API_BASE}/api/v1/schema/databases/${databaseId}/tables`);
    if (!response.ok) throw new Error('Failed to fetch tables');
    return response.json();
  },

  // Get detailed schema for a specific table
  getTableSchema: async (databaseId: string, tableName: string): Promise<TableSchema> => {
    const response = await fetch(`${API_BASE}/api/v1/schema/databases/${databaseId}/tables/${tableName}/schema`);
    if (!response.ok) throw new Error('Failed to fetch table schema');
    return response.json();
  },

  // Get sample data from table
  getSampleData: async (databaseId: string, tableName: string, limit = 5): Promise<SampleData> => {
    const response = await fetch(`${API_BASE}/api/v1/schema/databases/${databaseId}/tables/${tableName}/sample?limit=${limit}`);
    if (!response.ok) throw new Error('Failed to fetch sample data');
    return response.json();
  },

  // Generate AI description for table
  generateDescription: async (databaseId: string, tableName: string, regenerate = false): Promise<{ description: string; generatedAt: string; cached: boolean }> => {
    const response = await fetch(`${API_BASE}/api/v1/schema/databases/${databaseId}/tables/${tableName}/description`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ regenerate })
    });
    if (!response.ok) throw new Error('Failed to generate description');
    return response.json();
  }
};