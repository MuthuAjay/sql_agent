import React, { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Play, Save, Download, Code2, Loader2, AlertCircle, CheckCircle, History } from 'lucide-react';
import Editor from '@monaco-editor/react';
import { api, APIError } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { DataTable } from '@/components/data/DataTable';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { useTheme } from '@/contexts/ThemeContext';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import toast from 'react-hot-toast';

interface QueryTemplate {
  id: string;
  name: string;
  description: string;
  sql: string;
  category: string;
}

export function SQLEditorPage() {
  const { isDark } = useTheme();
  const [sql, setSql] = useState('-- Write your SQL query here\nSELECT * FROM users LIMIT 10;');
  const [savedQueries, setSavedQueries] = useLocalStorage<Array<{id: string; name: string; sql: string}>>('saved-sql-queries', []);
  const [queryResult, setQueryResult] = useState<any>(null);
  const [validationResult, setValidationResult] = useState<any>(null);

  const { data: templates, isLoading: templatesLoading } = useQuery({
    queryKey: ['sql-templates'],
    queryFn: api.sql.getTemplates,
    onError: (error) => {
      console.warn('Failed to load templates:', error);
    },
  });

  const executeMutation = useMutation({
    mutationFn: (sql: string) => api.sql.execute(sql),
    onSuccess: (response) => {
      setQueryResult(response);
      toast.success(`Query executed successfully! ${response.sql_result?.row_count || 0} rows returned`);
    },
    onError: (error: APIError) => {
      toast.error(`Query failed: ${error.message}`);
    },
  });

  const validateMutation = useMutation({
    mutationFn: api.query.validate,
    onSuccess: (result) => {
      setValidationResult(result);
      if (result.valid) {
        toast.success('Query is valid');
      } else {
        toast.error(`Query has ${result.errors.length} errors`);
      }
    },
    onError: (error: APIError) => {
      toast.error(`Validation failed: ${error.message}`);
    },
  });

  const handleExecute = () => {
    if (!sql.trim()) {
      toast.error('Please enter a SQL query');
      return;
    }
    executeMutation.mutate(sql);
  };

  const handleValidate = () => {
    if (!sql.trim()) {
      toast.error('Please enter a SQL query');
      return;
    }
    validateMutation.mutate(sql);
  };

  const handleSaveQuery = () => {
    if (!sql.trim()) {
      toast.error('Please enter a SQL query');
      return;
    }
    
    const name = prompt('Enter a name for this query:');
    if (name) {
      const newQuery = {
        id: Date.now().toString(),
        name,
        sql: sql.trim(),
      };
      setSavedQueries([...savedQueries, newQuery]);
      toast.success('Query saved successfully');
    }
  };

  const handleLoadTemplate = (template: QueryTemplate) => {
    setSql(template.sql);
    toast.success(`Loaded template: ${template.name}`);
  };

  const handleLoadSavedQuery = (query: {sql: string; name: string}) => {
    setSql(query.sql);
    toast.success(`Loaded query: ${query.name}`);
  };

  const handleExportResults = () => {
    if (!queryResult?.sql_result?.data) {
      toast.error('No results to export');
      return;
    }
    
    const csv = [
      queryResult.sql_result.columns.join(','),
      ...queryResult.sql_result.data.map((row: any) => 
        queryResult.sql_result.columns.map((col: string) => 
          JSON.stringify(row[col] ?? '')
        ).join(',')
      )
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `query-results-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-full bg-white dark:bg-gray-950 flex">
      {/* Left Panel - Editor */}
      <div className="flex-1 flex flex-col border-r border-gray-200 dark:border-gray-800">
        {/* Editor Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900">
          <div className="flex items-center space-x-2">
            <Code2 size={20} className="text-blue-600" />
            <h1 className="text-lg font-semibold text-gray-900 dark:text-white">SQL Editor</h1>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleValidate}
              loading={validateMutation.isPending}
            >
              {validateMutation.isPending ? <Loader2 size={16} /> : <CheckCircle size={16} />}
              Validate
            </Button>
            
            <Button
              variant="outline"
              size="sm"
              onClick={handleSaveQuery}
            >
              <Save size={16} />
              Save
            </Button>
            
            <Button
              onClick={handleExecute}
              loading={executeMutation.isPending}
            >
              {executeMutation.isPending ? <Loader2 size={16} /> : <Play size={16} />}
              Execute
            </Button>
          </div>
        </div>

        {/* Validation Results */}
        {validationResult && (
          <div className={`p-3 border-b ${
            validationResult.valid 
              ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
              : 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
          }`}>
            <div className="flex items-center space-x-2">
              {validationResult.valid ? (
                <CheckCircle size={16} className="text-green-600 dark:text-green-400" />
              ) : (
                <AlertCircle size={16} className="text-red-600 dark:text-red-400" />
              )}
              <span className={`text-sm font-medium ${
                validationResult.valid 
                  ? 'text-green-800 dark:text-green-200'
                  : 'text-red-800 dark:text-red-200'
              }`}>
                {validationResult.valid ? 'Query is valid' : `${validationResult.errors.length} errors found`}
              </span>
            </div>
            
            {validationResult.errors?.length > 0 && (
              <ul className="mt-2 text-sm text-red-700 dark:text-red-300 space-y-1">
                {validationResult.errors.map((error: string, index: number) => (
                  <li key={index}>• {error}</li>
                ))}
              </ul>
            )}
            
            {validationResult.warnings?.length > 0 && (
              <ul className="mt-2 text-sm text-yellow-700 dark:text-yellow-300 space-y-1">
                {validationResult.warnings.map((warning: string, index: number) => (
                  <li key={index}>⚠ {warning}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* SQL Editor */}
        <div className="flex-1">
          <Editor
            height="100%"
            defaultLanguage="sql"
            value={sql}
            onChange={(value) => setSql(value || '')}
            theme={isDark ? 'vs-dark' : 'light'}
            options={{
              minimap: { enabled: false },
              fontSize: 14,
              lineNumbers: 'on',
              roundedSelection: false,
              scrollBeyondLastLine: false,
              automaticLayout: true,
              tabSize: 2,
              wordWrap: 'on',
            }}
          />
        </div>

        {/* Query Results */}
        {queryResult && (
          <div className="border-t border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900 p-4">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-400">
                <span>Execution time: {queryResult.sql_result?.execution_time}ms</span>
                <span>Rows: {queryResult.sql_result?.row_count}</span>
              </div>
              
              {queryResult.sql_result?.data?.length > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleExportResults}
                >
                  <Download size={16} />
                  Export CSV
                </Button>
              )}
            </div>

            {queryResult.sql_result?.data?.length > 0 ? (
              <div className="max-h-64 overflow-hidden">
                <DataTable
                  data={queryResult.sql_result.data}
                  columns={queryResult.sql_result.columns}
                  totalRows={queryResult.sql_result.total_rows}
                />
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                No results returned
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right Panel - Templates & History */}
      <div className="w-80 bg-gray-50 dark:bg-gray-900 border-l border-gray-200 dark:border-gray-800 flex flex-col">
        {/* Templates */}
        <Card className="m-4 flex-1">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center space-x-2">
              <Code2 size={18} />
              <span>Query Templates</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="overflow-auto max-h-64">
            {templatesLoading ? (
              <div className="flex justify-center py-4">
                <LoadingSpinner size="sm" />
              </div>
            ) : templates && templates.length > 0 ? (
              <div className="space-y-2">
                {templates.map((template) => (
                  <div
                    key={template.id}
                    className="p-3 bg-white dark:bg-gray-800 rounded-md border border-gray-200 dark:border-gray-700 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                    onClick={() => handleLoadTemplate(template)}
                  >
                    <div className="font-medium text-sm text-gray-900 dark:text-white mb-1">
                      {template.name}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                      {template.description}
                    </div>
                    <div className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 rounded">
                      {template.category}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 dark:text-gray-400 text-center py-4 text-sm">
                No templates available
              </p>
            )}
          </CardContent>
        </Card>

        {/* Saved Queries */}
        <Card className="m-4 flex-1">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center space-x-2">
              <History size={18} />
              <span>Saved Queries</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="overflow-auto max-h-64">
            {savedQueries.length > 0 ? (
              <div className="space-y-2">
                {savedQueries.map((query) => (
                  <div
                    key={query.id}
                    className="p-3 bg-white dark:bg-gray-800 rounded-md border border-gray-200 dark:border-gray-700 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                    onClick={() => handleLoadSavedQuery(query)}
                  >
                    <div className="font-medium text-sm text-gray-900 dark:text-white mb-1">
                      {query.name}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 truncate">
                      {query.sql}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 dark:text-gray-400 text-center py-4 text-sm">
                No saved queries
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}