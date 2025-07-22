// API Response Types
export interface QueryResponse {
  request_id: string;
  timestamp: string;
  processing_time: number;
  query: string;
  intent: "sql_generation" | "analysis" | "visualization" | "schema_info";
  confidence: number;
  sql_result?: SQLResult;
  analysis_result?: AnalysisResult;
  visualization_result?: VisualizationResult;
  suggestions: string[];
  cached: boolean;
}

export interface SQLResult {
  sql: string;
  data: Array<Record<string, any>>;
  row_count: number;
  total_rows?: number;
  execution_time: number;
  columns: string[];
  column_types?: Record<string, string>;
  explanation?: string;
  warnings: string[];
}

export interface AnalysisResult {
  summary: StatisticalSummary;
  insights: Insight[];
  anomalies: Anomaly[];
  trends: Trend[];
  recommendations: Recommendation[];
  data_quality_score: number;
  confidence_score: number;
}

export interface VisualizationResult {
  chart_type: string;
  title: string;
  data: ChartData;
  config: ChartConfig;
  insights: string[];
}

export interface StatisticalSummary {
  total_rows: number;
  numeric_columns: ColumnStats[];
  categorical_columns: CategoryStats[];
  missing_values: Record<string, number>;
  data_quality_issues: string[];
}

export interface Insight {
  type: string;
  title: string;
  description: string;
  confidence: number;
  impact: 'high' | 'medium' | 'low';
  data_points: any[];
}

export interface Anomaly {
  column: string;
  type: string;
  severity: 'high' | 'medium' | 'low';
  description: string;
  affected_rows: number;
}

export interface Trend {
  column: string;
  trend_type: 'increasing' | 'decreasing' | 'stable' | 'seasonal';
  confidence: number;
  description: string;
  chart_data: any[];
}

export interface Recommendation {
  type: string;
  title: string;
  description: string;
  priority: 'high' | 'medium' | 'low';
  action: string;
  expected_impact: string;
}

export interface SchemaResponse {
  database_name: string;
  tables: TableInfo[];
  relationships: RelationshipInfo[];
  total_tables: number;
  total_columns: number;
  last_updated: string;
}

export interface TableInfo {
  name: string;
  columns: ColumnInfo[];
  row_count?: number;
  size?: string;
  description?: string;
  relationships: RelationshipInfo[];
}

export interface ColumnInfo {
  name: string;
  type: string;
  nullable: boolean;
  primary_key: boolean;
  foreign_key?: ForeignKeyInfo;
  description?: string;
}

export interface RelationshipInfo {
  from_table: string;
  from_column: string;
  to_table: string;
  to_column: string;
  relationship_type: 'one_to_one' | 'one_to_many' | 'many_to_many';
}

export interface ForeignKeyInfo {
  referenced_table: string;
  referenced_column: string;
}

export interface ColumnStats {
  column: string;
  min: number;
  max: number;
  mean: number;
  median: number;
  std_dev: number;
  null_count: number;
}

export interface CategoryStats {
  column: string;
  unique_values: number;
  top_values: Array<{ value: string; count: number }>;
  null_count: number;
}

export interface ChartData {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string;
    borderWidth?: number;
  }>;
}

export interface ChartConfig {
  type: 'bar' | 'line' | 'pie' | 'scatter' | 'area';
  options: any;
  plugins?: any[];
}

export interface ErrorResponse {
  error: {
    type: string;
    status_code: number;
    detail: string;
    user_message: string;
    suggestions: string[];
    request_id: string;
    timestamp: number;
  };
}

export interface QueryOptions {
  database?: string;
  includeAnalysis?: boolean;
  includeVisualization?: boolean;
  maxResults?: number;
  sessionId?: string;
}

export interface PerformanceTestResult {
  total_time: number;
  successful: number;
  failed: number;
  queries_per_second: number;
  avg_response_time: number;
  error_details: string[];
}

export interface HealthStatus {
  api_status: 'healthy' | 'degraded' | 'down';
  database_status: 'connected' | 'disconnected' | 'error';
  response_time: number;
  last_check: string;
  error_count: number;
  uptime: number;
}