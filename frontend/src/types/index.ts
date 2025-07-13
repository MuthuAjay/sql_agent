// frontend/src/types/index.ts

// Base types
export interface BaseResponse {
  request_id: string;
  timestamp: string;
  processing_time: number;
}

// Query Types
export interface QueryRequest {
  query: string;
  database_name?: string;
  max_results?: number;
  include_analysis?: boolean;
  include_visualization?: boolean;
  chart_type?: string;
  analysis_type?: string;
  context?: Record<string, any>;
  session_id?: string;
}

export interface QueryResponse extends BaseResponse {
  query: string;
  intent: 'sql_generation' | 'analysis' | 'visualization' | 'schema_info' | 'unknown';
  confidence: number;
  sql_result?: SQLResult;
  analysis_result?: AnalysisResult;
  visualization_result?: VisualizationResult;
  suggestions: string[];
  cached: boolean;
}

// SQL Types
export interface SQLResult {
  sql: string;
  data: any[][];
  row_count: number;
  total_rows?: number;
  execution_time: number;
  columns: string[];
  column_types?: Record<string, string>;
  explanation?: string;
  query_plan?: Record<string, any>;
  cache_hit: boolean;
  warnings: string[];
  error?: string;
}

export interface ValidationResult {
  is_valid: boolean;
  errors: string[];
  warnings: string[];
  suggestions: string[];
  estimated_cost?: number;
  estimated_rows?: number;
}

// Analysis Types
export interface StatisticalSummary {
  count: number;
  numeric_columns: Record<string, Record<string, number>>;
  categorical_columns: Record<string, Record<string, any>>;
  missing_values: Record<string, number>;
  data_types: Record<string, string>;
  correlations?: Record<string, Record<string, number>>;
}

export interface Insight {
  type: string;
  title: string;
  description: string;
  confidence: number;
  impact: string;
  supporting_data?: Record<string, any>;
}

export interface Anomaly {
  type: string;
  column?: string;
  description: string;
  severity: string;
  affected_rows?: number;
  threshold?: number;
}

export interface Trend {
  type: string;
  column: string;
  direction: string;
  strength: number;
  period?: string;
  description: string;
}

export interface Recommendation {
  type: string;
  title: string;
  description: string;
  priority: string;
  effort: string;
  expected_impact: string;
  action_items: string[];
}

export interface AnalysisResult {
  summary: StatisticalSummary;
  insights: Insight[];
  anomalies: Anomaly[];
  trends: Trend[];
  recommendations: Recommendation[];
  data_quality_score: number;
  confidence_score: number;
  processing_metadata?: Record<string, any>;
}

// Visualization Types
export interface ChartConfig {
  type: string;
  title: string;
  x_axis?: string;
  y_axis?: string;
  color_by?: string;
  size_by?: string;
  aggregation?: string;
  theme: string;
  interactive: boolean;
  responsive: boolean;
  animations: boolean;
  legend: boolean;
  grid: boolean;
}

export interface VisualizationResult {
  chart_type: string;
  chart_config: ChartConfig;
  chart_data: Record<string, any>;
  title: string;
  description?: string;
  export_formats: string[];
  alternative_charts: string[];
  data_insights: string[];
}

export interface VisualizationSuggestion {
  type: string;
  title: string;
  description: string;
  confidence: number;
  columns: string[];
  config: Record<string, any>;
}

// Database and Schema Types
export interface DatabaseInfo {
  id: string;
  name: string;
  type: 'postgresql' | 'mysql' | 'sqlite' | 'mongodb' | 'mssql' | 'oracle' | 'bigquery' | 'snowflake';
  status: 'connected' | 'disconnected' | 'connecting' | 'error';
  host?: string;
  port?: number;
  version?: string;
  size?: string;
  table_count?: number;
  last_sync: string;
  capabilities: string[];
  connection_pool_size?: number;
}

export interface ColumnInfo {
  name: string;
  type: string;
  nullable: boolean;
  primary_key: boolean;
  foreign_key?: string;
  unique: boolean;
  indexed: boolean;
  default_value?: any;
  max_length?: number;
  precision?: number;
  scale?: number;
  description?: string;
  sample_values?: any[];
}

export interface IndexInfo {
  name: string;
  columns: string[];
  unique: boolean;
  primary: boolean;
  type: string;
  size?: string;
}

export interface RelationshipInfo {
  type: string;
  source_table: string;
  source_column: string;
  target_table: string;
  target_column: string;
  constraint_name?: string;
  on_delete?: string;
  on_update?: string;
}

export interface TableInfo {
  id: string;
  name: string;
  database_id: string;
  schema?: string;
  columns: ColumnInfo[];
  indexes: IndexInfo[];
  relationships: RelationshipInfo[];
  row_count?: number;
  size?: string;
  description?: string;
  created_at?: string;
  updated_at?: string;
  primary_keys: string[];
  foreign_keys: string[];
}

// Query History Types
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

// Connection Types
export interface Connection {
  id: string;
  name: string;
  type: 'postgresql' | 'mysql' | 'sqlite' | 'mssql' | 'oracle' | 'mongodb';
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

// User Types
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

// Chat Types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  sql?: string;
  query_result?: SQLResult;
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
  items: T[];
  total: number;
  page: number;
  limit: number;
  pages: number;
  has_next: boolean;
  has_prev: boolean;
}

// Error Types
export interface ApiError {
  type: string;
  status_code: number;
  detail: string;
  request_id: string;
  timestamp?: string;
  suggestions?: string[];
}

// Feature Flag Types
export interface FeatureFlags {
  query_optimization: boolean;
  sql_validation: boolean;
  visualization: boolean;
  analysis: boolean;
  websockets: boolean;
  query_caching: boolean;
  metrics: boolean;
  rag: boolean;
}

// App Configuration Types
export interface AppConfig {
  app_name: string;
  app_version: string;
  environment: string;
  api_prefix: string;
  features: FeatureFlags;
  limits: {
    max_rows_returned: number;
    query_timeout_seconds: number;
    max_file_size_mb: number;
    allowed_file_types: string[];
  };
  llm_provider: string;
}

// Health Check Types
export interface HealthStatus {
  status: string;
  timestamp: number;
  version: string;
  services: Record<string, string>;
  metrics?: Record<string, any>;
  uptime?: number;
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

// State Management Types
export interface AppState {
  user: {
    profile: UserProfile | null;
    preferences: UserPreferences;
    isAuthenticated: boolean;
    loading: boolean;
  };
  databases: {
    list: DatabaseInfo[];
    current: DatabaseInfo | null;
    loading: boolean;
    error: string | null;
  };
  queries: {
    history: QueryHistory[];
    current: QueryResponse | null;
    loading: boolean;
    error: string | null;
  };
  schema: {
    tables: TableInfo[];
    currentTable: TableInfo | null;
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
  onMutate?: (variables: any) => any;
}

// Backward compatibility - Legacy types
export interface Database extends DatabaseInfo {}
export interface Table extends TableInfo {}
export interface QueryResult extends SQLResult {}