// frontend/src/hooks/useApi.ts
import { useState, useEffect, useCallback, useRef } from 'react';
import { useQueryClient, useMutation, useQuery } from '@tanstack/react-query';
import apiClient from '../services/api';
import type {
  QueryRequest,
  QueryResponse,
  SQLResult,
  AnalysisResult,
  VisualizationResult,
  ValidationResult,
  DatabaseInfo,
  TableInfo,
  QueryHistory
} from '../types/index';

// Query Keys for React Query
export const QUERY_KEYS = {
  DATABASES: ['databases'],
  TABLES: (databaseId: string) => ['tables', databaseId],
  TABLE_DETAILS: (databaseId: string, tableName: string) => ['table', databaseId, tableName],
  SAMPLE_DATA: (databaseId: string, tableName: string) => ['sample', databaseId, tableName],
  QUERY_HISTORY: ['queryHistory'],
  HEALTH: ['health'],
} as const;

// Generic API hook with enhanced error handling
interface UseApiOptions<T> {
  enabled?: boolean;
  staleTime?: number;
  cacheTime?: number;
  retry?: number;
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
}

export function useApi<T>(
  apiCall: () => Promise<T>,
  dependencies: any[] = [],
  options: UseApiOptions<T> = {}
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const execute = useCallback(async () => {
    // Cancel previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setLoading(true);
    setError(null);

    try {
      const result = await apiCall();
      
      if (!abortControllerRef.current?.signal.aborted) {
        setData(result);
        options.onSuccess?.(result);
      }
    } catch (err) {
      if (!abortControllerRef.current?.signal.aborted) {
        const error = err instanceof Error ? err : new Error('Unknown error');
        setError(error);
        options.onError?.(error);
      }
    } finally {
      if (!abortControllerRef.current?.signal.aborted) {
        setLoading(false);
      }
    }
  }, [apiCall, options]);

  useEffect(() => {
    if (options.enabled !== false) {
      execute();
    }

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, dependencies);

  return {
    data,
    loading,
    error,
    execute,
    refetch: execute
  };
}

// Enhanced mutation hook
export function useMutation<T, P = any>(
  mutationFn: (params: P) => Promise<T>,
  options: {
    onSuccess?: (data: T, params: P) => void;
    onError?: (error: Error, params: P) => void;
    onSettled?: (data: T | null, error: Error | null, params: P) => void;
  } = {}
) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const mutate = useCallback(async (params: P): Promise<T> => {
    setLoading(true);
    setError(null);

    try {
      const result = await mutationFn(params);
      options.onSuccess?.(result, params);
      options.onSettled?.(result, null, params);
      return result;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      options.onError?.(error, params);
      options.onSettled?.(null, error, params);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [mutationFn, options]);

  return {
    mutate,
    loading,
    error,
    reset: () => setError(null)
  };
}

// Query processing hooks
export function useQueryProcessor() {
  const [currentStep, setCurrentStep] = useState<string>('');
  const abortControllerRef = useRef<AbortController | null>(null);

  const processQuery = useMutation<QueryResponse, QueryRequest>({
    mutationFn: async (request: QueryRequest) => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      
      abortControllerRef.current = new AbortController();
      setCurrentStep('Processing query...');

      try {
        const result = await apiClient.query.naturalLanguage(
          request.query, 
          request.database_name || 'default'
        );
        setCurrentStep('Complete');
        return result;
      } catch (error) {
        setCurrentStep('Error');
        throw error;
      }
    }
  });

  const processSimpleQuery = useMutation<any, QueryRequest>({
    mutationFn: (request: QueryRequest) => {
      return apiClient.query.processSimple(request);
    }
  });

  const validateQuery = useMutation<ValidationResult, { query: string; databaseId: string }>({
    mutationFn: ({ query, databaseId }) => apiClient.query.validate(query, databaseId)
  });

  const cancelQuery = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setCurrentStep('Cancelled');
    }
  }, []);

  return {
    processQuery: processQuery.mutate,
    processSimpleQuery: processSimpleQuery.mutate,
    validateQuery: validateQuery.mutate,
    cancelQuery,
    
    // Status
    isProcessing: processQuery.loading,
    isValidating: validateQuery.loading,
    error: processQuery.error || validateQuery.error,
    
    // Results
    queryResult: processQuery.data,
    simpleResult: processSimpleQuery.data,
    validationResult: validateQuery.data,
    
    // Processing state
    currentStep,
    
    // Reset
    reset: () => {
      processQuery.reset();
      processSimpleQuery.reset();
      validateQuery.reset();
      setCurrentStep('');
    }
  };
}

// SQL operations hook
export function useSQLOperations() {
  const executeSQL = useMutation<SQLResult, { sql: string; databaseId: string; dryRun?: boolean }>({
    mutationFn: ({ sql, databaseId, dryRun = false }) => 
      apiClient.sql.execute(sql, databaseId, dryRun)
  });

  const validateSQL = useMutation<any, { sql: string; databaseId: string }>({
    mutationFn: ({ sql, databaseId }) => apiClient.sql.validate(sql, databaseId)
  });

  const formatSQL = useMutation<{ formatted: string }, { sql: string }>({
    mutationFn: ({ sql }) => apiClient.sql.format(sql)
  });

  const optimizeSQL = useMutation<{ suggestions: string[] }, { sql: string; databaseId: string }>({
    mutationFn: ({ sql, databaseId }) => apiClient.sql.optimize(sql, databaseId)
  });

  return {
    executeSQL: executeSQL.mutate,
    validateSQL: validateSQL.mutate,
    formatSQL: formatSQL.mutate,
    optimizeSQL: optimizeSQL.mutate,
    
    // Status
    isExecuting: executeSQL.loading,
    isValidating: validateSQL.loading,
    isFormatting: formatSQL.loading,
    isOptimizing: optimizeSQL.loading,
    
    // Results
    executionResult: executeSQL.data,
    validationResult: validateSQL.data,
    formattingResult: formatSQL.data,
    optimizationResult: optimizeSQL.data,
    
    // Errors
    executeError: executeSQL.error,
    validateError: validateSQL.error,
    formatError: formatSQL.error,
    optimizeError: optimizeSQL.error,
    
    // Reset
    reset: () => {
      executeSQL.reset();
      validateSQL.reset();
      formatSQL.reset();
      optimizeSQL.reset();
    }
  };
}

// Analysis operations hook
export function useAnalysisOperations() {
  const analyzeData = useMutation<AnalysisResult, {
    sql: string;
    databaseId: string;
    analysisType?: string;
  }>({
    mutationFn: ({ sql, databaseId, analysisType = 'comprehensive' }) =>
      apiClient.analysis.summarize(sql, databaseId, analysisType)
  });

  const profileData = useMutation<any, {
    table: string;
    columns: string[];
    databaseId: string;
  }>({
    mutationFn: ({ table, columns, databaseId }) =>
      apiClient.analysis.profile(table, columns, databaseId)
  });

  return {
    analyzeData: analyzeData.mutate,
    profileData: profileData.mutate,
    
    // Status
    isAnalyzing: analyzeData.loading || profileData.loading,
    
    // Results
    analysisResult: analyzeData.data,
    profileResult: profileData.data,
    
    // Errors
    analysisError: analyzeData.error,
    profileError: profileData.error,
    
    // Reset
    reset: () => {
      analyzeData.reset();
      profileData.reset();
    }
  };
}

// Visualization operations hook
export function useVisualizationOperations() {
  const suggestVisualizations = useMutation<{ suggestions: VisualizationResult[] }, {
    sql: string;
    databaseId: string;
  }>({
    mutationFn: ({ sql, databaseId }) =>
      apiClient.visualization.suggest(sql, databaseId)
  });

  const generateVisualization = useMutation<any, {
    data: any[];
    chartType: string;
    config: any;
  }>({
    mutationFn: ({ data, chartType, config }) =>
      apiClient.visualization.generate(data, chartType, config)
  });

  return {
    suggestVisualizations: suggestVisualizations.mutate,
    generateVisualization: generateVisualization.mutate,
    
    // Status
    isSuggesting: suggestVisualizations.loading,
    isGenerating: generateVisualization.loading,
    
    // Results
    suggestions: suggestVisualizations.data,
    visualization: generateVisualization.data,
    
    // Errors
    suggestError: suggestVisualizations.error,
    generateError: generateVisualization.error,
    
    // Reset
    reset: () => {
      suggestVisualizations.reset();
      generateVisualization.reset();
    }
  };
}

// Database and schema hooks
export function useDatabases() {
  return useQuery({
    queryKey: QUERY_KEYS.DATABASES,
    queryFn: () => apiClient.schema.getDatabases(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

export function useTables(databaseId: string | null) {
  return useQuery({
    queryKey: QUERY_KEYS.TABLES(databaseId || ''),
    queryFn: () => databaseId ? apiClient.schema.getTables(databaseId) : Promise.resolve({ tables: [] }),
    enabled: !!databaseId,
    staleTime: 5 * 60 * 1000,
  });
}

export function useTableDetails(databaseId: string | null, tableName: string | null) {
  return useQuery({
    queryKey: QUERY_KEYS.TABLE_DETAILS(databaseId || '', tableName || ''),
    queryFn: () => 
      databaseId && tableName 
        ? apiClient.schema.getTable(databaseId, tableName)
        : Promise.resolve(null),
    enabled: !!(databaseId && tableName),
    staleTime: 10 * 60 * 1000,
  });
}

export function useSampleData(databaseId: string | null, tableName: string | null, limit = 5) {
  return useQuery({
    queryKey: QUERY_KEYS.SAMPLE_DATA(databaseId || '', tableName || ''),
    queryFn: () =>
      databaseId && tableName
        ? apiClient.schema.getSampleData(databaseId, tableName, limit)
        : Promise.resolve(null),
    enabled: !!(databaseId && tableName),
    staleTime: 5 * 60 * 1000,
  });
}

// Query history hook
export function useQueryHistory(page = 1, limit = 20, databaseId?: string) {
  return useQuery({
    queryKey: [...QUERY_KEYS.QUERY_HISTORY, { page, limit, databaseId }],
    queryFn: () => apiClient.query.getHistory(page, limit, databaseId),
    keepPreviousData: true,
    staleTime: 30 * 1000, // 30 seconds
  });
}

// Schema search hook
export function useSchemaSearch(databaseId: string | null) {
  const searchSchema = useMutation<any[], { query: string; types?: string }>({
    mutationFn: ({ query, types = 'table,column' }) =>
      databaseId 
        ? apiClient.schema.searchSchema(databaseId, query, types)
        : Promise.resolve([])
  });

  return {
    searchSchema: searchSchema.mutate,
    isSearching: searchSchema.loading,
    searchResults: searchSchema.data,
    searchError: searchSchema.error,
    resetSearch: searchSchema.reset
  };
}

// Health check hook
export function useHealthCheck() {
  return useQuery({
    queryKey: QUERY_KEYS.HEALTH,
    queryFn: () => apiClient.health.check(),
    refetchInterval: 30 * 1000, // Every 30 seconds
    retry: false,
  });
}

// Debounced API call hook
export function useDebouncedApi<T, P>(
  apiCall: (params: P) => Promise<T>,
  delay = 500
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const timeoutRef = useRef<NodeJS.Timeout>();
  const abortControllerRef = useRef<AbortController>();

  const execute = useCallback((params: P) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    timeoutRef.current = setTimeout(async () => {
      abortControllerRef.current = new AbortController();
      setLoading(true);
      setError(null);

      try {
        const result = await apiCall(params);
        
        if (!abortControllerRef.current?.signal.aborted) {
          setData(result);
        }
      } catch (err) {
        if (!abortControllerRef.current?.signal.aborted) {
          const error = err instanceof Error ? err : new Error('Unknown error');
          setError(error);
        }
      } finally {
        if (!abortControllerRef.current?.signal.aborted) {
          setLoading(false);
        }
      }
    }, delay);
  }, [apiCall, delay]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    data,
    loading,
    error,
    execute
  };
}

// Connection testing hook
export function useConnectionTest() {
  const testConnection = useMutation<{ success: boolean; message?: string }, { connectionId: string }>({
    mutationFn: ({ connectionId }) => apiClient.connection.testConnection(connectionId)
  });

  return {
    testConnection: testConnection.mutate,
    isTesting: testConnection.loading,
    testResult: testConnection.data,
    testError: testConnection.error,
    resetTest: testConnection.reset
  };
}

// Export all hooks
export {
  useApi,
  useMutation,
  useQueryProcessor,
  useSQLOperations,
  useAnalysisOperations,
  useVisualizationOperations,
  useDatabases,
  useTables,
  useTableDetails,
  useSampleData,
  useQueryHistory,
  useSchemaSearch,
  useHealthCheck,
  useDebouncedApi,
  useConnectionTest
};