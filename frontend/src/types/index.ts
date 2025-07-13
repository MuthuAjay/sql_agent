// frontend/src/types/index.ts

// Core Database Types
export interface Database {
  id: string;
  name: string;
  type: 'postgres' | 'mysql' | 'sqlite' | 'mssql' | 'oracle';
  host?: string;
  port?: number;
  status: 'connected' | 'disconnected' | 'error';
  description?: string;
  created_at: string;
  updated_at: string;
}

export interface Table {
  id: string;
  name: string;
  database_id: string;
  schema?: string;
  type: 'table' | 'view' | 'materialized_view';
  row_count?: number;
  size?: string;
  description?: string;
  created_at?: string;
  updated_at?: string;
}

export interface Column {
  name: string;
  type: string;
  nullable: boolean;
  default_value?: any;
  is_primary_key: boolean;
  is_foreign_key: boolean;
  foreign_key_table?: string;
  foreign_key_column?: string;
  max_length?: number;
  precision?: number;
  scale?: number;
  description?: string;
}

export interface Index {
  name: string;
  columns: string[];
  is_unique: boolean;
  is_primary: boolean;
  type: string;
}

export interface Relationship {
  type: 'one_to_one' | 'one_to_many' | 'many_to_many';
  source_table: string;
  source_column: string;
  target_table: string;
  target_column: string;
  constraint_name?: string;
}

export interface TableSchema {
  table: Table;
  columns: Column[];
  indexes: Index[];
  relationships: Relationship[];
  constraints: any[];
  triggers: any[];
  row_count: number;
  size_bytes: number;
}

export interface SampleData {
  columns: string[];
  data: any[][];
  total_rows: number;
  sample_size: number;
}

// Query Types
export interface QueryResult {
  id: string;
  sql: string;
  status: 'success' | 'error' | 'running' | 'cancelled';
  columns?: string[];
  data?: any[][];
  row_count?: number;
  execution_time?: number;
  error_message?: string;
  database_id: string;
  created_at: string;
  metadata?: {
    affected_rows?: number;
    execution_plan?: any;
    warnings?: string[];
  };
}

export interface QueryHistory {
  id: string;
  query: string;
  sql?: string;
  database_id: string;
  database_name: string;
  status: 'success' | 'error' | 'cancelled';
  execution_time?: number;
  row_count?: number;
  error_message?: string;
  created_at: string;
  user_id?: string;
  is_favorite: boolean;
}

export interface ValidationResult {
  valid: boolean;
  suggestions?: string[];
  errors?: string[];
  warnings?: string[];
}

export interface NaturalLanguageQuery {
  id: string;
  original_query: string;
  generated_sql: string;
  confidence: number;
  explanation?: string;
  suggestions?: string[];
}

// Analysis Types
export interface DataProfile {
  column: string;
  type: string;
  count: number;
  null_count: number;
  unique_count: number;
  min_value?: any;
  max_value?: any;
  mean?: number;
  median?: number;
  std_dev?: number;
  percentiles?: { [key: string]: number };
  most_frequent?: any[];
  data_quality_score: number;
}

export interface DataSummary {
  total_rows: number;
  total_columns: number;
  data_types: { [key: string]: number };
  null_percentage: number;
  duplicate_rows: number;
  data_quality_score: number;
  insights: string[];
  recommendations: string[];
}

// Visualization Types
export interface VisualizationSuggestion {
  type: 'bar' | 'line' | 'pie' | 'scatter' | 'histogram' | 'heatmap' | 'table';
  title: string;
  description: string;
  confidence: number;
  columns: string[];
  config: {
    x_axis?: string;
    y_axis?: string | string[];
    color?: string;
    size?: string;
    aggregation?: 'sum' | 'count' | 'avg' | 'min' | 'max';
  };
}

export interface ChartConfig {
  type: string;
  title: string;
  x_axis: string;
  y_axis: string | string[];
  color?: string;
  size?: string;
  aggregation?: string;
  theme?: 'light' | 'dark';
  responsive?: boolean;
  animation?: boolean;
  legend?: boolean;
  grid?: boolean;
}

// Connection Types
export interface Connection {
  id: string;
  name: string;
  type: 'postgres' | 'mysql' | 'sqlite' | 'mssql' | 'oracle' | 'bigquery' | 'snowflake';
  host: string;
  port: number;
  database: string;
  username: string;
  password?: string;
  ssl?: boolean;
  status: 'connected' | 'disconnected' | 'error' | 'testing';
  last_tested?: string;
  created_at: string;
  updated_at: string;
  description?: string;
  tags?: string[];
}

// Chat Types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  sql?: string;
  query_result?: QueryResult;
  suggestions?: string[];
  context?: any;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  database_id?: string;
  created_at: string;
  updated_at: string;
}

// User Types
export interface UserProfile {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  role: 'admin' | 'analyst' | 'viewer';
  created_at: string;
  last_login?: string;
  preferences: UserPreferences;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  language: string;
  timezone: string;
  date_format: string;
  number_format: string;
  default_database?: string;
  query_timeout: number;
  auto_save: boolean;
  show_query_hints: boolean;
  enable_ai_suggestions: boolean;
  chart_preferences: {
    default_theme: 'light' | 'dark';
    animation: boolean;
    responsive: boolean;
  };
  editor_preferences: {
    theme: 'light' | 'dark';
    font_size: number;
    word_wrap: boolean;
    show_line_numbers: boolean;
    auto_complete: boolean;
  };
}

// API Response Types
export interface ApiResponse<T = any> {
  data: T;
  status: 'success' | 'error';
  message?: string;
  errors?: string[];
  metadata?: {
    total?: number;
    page?: number;
    limit?: number;
    pages?: number;
  };
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}

// Error Types
export interface ApiError {
  type: string;
  status_code: number;
  detail: string;
  request_id: string;
  timestamp?: string;
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'query_progress' | 'query_complete' | 'query_error' | 'notification';
  data: any;
  request_id?: string;
  timestamp: string;
}

// Export Types
export interface ExportOptions {
  format: 'csv' | 'json' | 'xlsx' | 'parquet';
  include_headers: boolean;
  max_rows?: number;
  compression?: boolean;
}

// Search Types
export interface SearchResult {
  type: 'table' | 'column' | 'view' | 'procedure';
  name: string;
  table_name?: string;
  description?: string;
  match_score: number;
  database_id: string;
}

// Notification Types
export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  action?: {
    label: string;
    url: string;
  };
}

// Dashboard Types
export interface Widget {
  id: string;
  type: 'chart' | 'table' | 'metric' | 'text';
  title: string;
  sql: string;
  database_id: string;
  config: any;
  position: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  refresh_interval?: number;
  last_updated?: string;
}

export interface Dashboard {
  id: string;
  name: string;
  description?: string;
  widgets: Widget[];
  layout: any;
  shared: boolean;
  created_at: string;
  updated_at: string;
  created_by: string;
  tags?: string[];
}

// Report Types
export interface Report {
  id: string;
  name: string;
  description?: string;
  sql: string;
  database_id: string;
  parameters?: ReportParameter[];
  schedule?: ReportSchedule;
  format: 'pdf' | 'html' | 'csv' | 'xlsx';
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface ReportParameter {
  name: string;
  type: 'string' | 'number' | 'date' | 'boolean';
  required: boolean;
  default_value?: any;
  description?: string;
}

export interface ReportSchedule {
  enabled: boolean;
  frequency: 'daily' | 'weekly' | 'monthly';
  time: string;
  recipients: string[];
  next_run?: string;
}

// State Management Types
export interface AppState {
  user: {
    profile: UserProfile | null;
    preferences: UserPreferences;
    isAuthenticated: boolean;
    loading: boolean;
  };
  databases: {
    list: Database[];
    current: Database | null;
    loading: boolean;
    error: string | null;
  };
  queries: {
    history: QueryHistory[];
    current: QueryResult | null;
    loading: boolean;
    error: string | null;
  };
  schema: {
    tables: Table[];
    currentTable: TableSchema | null;
    loading: boolean;
    error: string | null;
  };
  chat: {
    sessions: ChatSession[];
    currentSession: ChatSession | null;
    loading: boolean;
    error: string | null;
  };
  notifications: Notification[];
}

// Hook Types
export interface UseQueryOptions {
  enabled?: boolean;
  refetchOnWindowFocus?: boolean;
  retry?: number;
  retryDelay?: number;
}

export interface UseMutationOptions<T = any> {
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
  onSettled?: () => void;
}