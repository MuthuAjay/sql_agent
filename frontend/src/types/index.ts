// Core types for the SQL Agent application
export interface Database {
  id: string;
  name: string;
  type: 'postgresql' | 'mysql' | 'sqlite' | 'mongodb';
  status: 'connected' | 'disconnected' | 'connecting' | 'error';
  lastSync: string;
}

export interface Table {
  name: string;
  schema: string;
  rowCount: number;
  size: string;
  columns: Column[];
}

export interface Column {
  name: string;
  type: string;
  nullable: boolean;
  primaryKey?: boolean;
  foreignKey?: boolean;
}

export interface QueryResult {
  requestId: string;
  sqlQuery: string;
  results: any[];
  executionTime: number;
  rowCount: number;
  explanation?: string;
  error?: string;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
  queryId?: string;
}

export interface QueryHistory {
  id: string;
  query: string;
  sqlQuery: string;
  timestamp: string;
  databaseId: string;
  executionTime: number;
  rowCount: number;
}

export interface VisualizationSuggestion {
  type: 'bar_chart' | 'line_chart' | 'pie_chart' | 'scatter_plot' | 'histogram';
  confidence: number;
  config: {
    xAxis: string;
    yAxis: string;
    title: string;
    groupBy?: string;
  };
}

export interface Connection {
  id: string;
  name: string;
  type: string;
  host: string;
  port: number;
  database: string;
  username: string;
  sslMode?: string;
  status: 'connected' | 'disconnected' | 'testing';
}

export interface UserPreferences {
  defaultDatabase?: string;
  queryTimeout: number;
  autoLimit: number;
  preferredViz: string;
  theme: 'light' | 'dark';
}