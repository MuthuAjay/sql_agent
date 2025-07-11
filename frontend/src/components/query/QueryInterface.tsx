import React, { useState } from 'react';
import { NaturalLanguageInput } from './NaturalLanguageInput';
import { SqlEditor } from './SqlEditor';
import { QueryHistory } from './QueryHistory';
import { ResultsTable } from './ResultsTable';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../common/Tabs';
import { Play, Save, History, Code, MessageSquare } from 'lucide-react';
import { useQueryStore, useUIStore } from '../../stores';
import { queryAPI, sqlAPI } from '../../services/api';
import { LoadingSpinner } from '../common/LoadingSpinner';
import toast from 'react-hot-toast';

export const QueryInterface: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'natural' | 'sql'>('natural');
  const { 
    currentQuery, 
    currentSql, 
    queryResult, 
    isExecuting,
    setCurrentQuery,
    setCurrentSql,
    setQueryResult,
    setIsExecuting,
    setError,
    addToHistory
  } = useQueryStore();
  const { activeDatabase } = useUIStore();

  const handleExecuteNaturalLanguage = async () => {
    if (!currentQuery.trim() || !activeDatabase) {
      toast.error('Please enter a query and select a database');
      return;
    }

    setIsExecuting(true);
    setError(null);

    try {
      const result = await queryAPI.naturalLanguage(currentQuery, activeDatabase);
      setQueryResult(result);
      setCurrentSql(result.sqlQuery);
      
      // Add to history
      addToHistory({
        id: result.requestId,
        query: currentQuery,
        sqlQuery: result.sqlQuery,
        timestamp: new Date().toISOString(),
        databaseId: activeDatabase,
        executionTime: result.executionTime,
        rowCount: result.rowCount,
      });

      toast.success(`Query executed successfully! ${result.rowCount} rows returned`);
    } catch (error: any) {
      setError(error.response?.data?.error?.detail || 'Failed to execute query');
      toast.error('Failed to execute query');
    } finally {
      setIsExecuting(false);
    }
  };

  const handleExecuteSQL = async () => {
    if (!currentSql.trim() || !activeDatabase) {
      toast.error('Please enter SQL and select a database');
      return;
    }

    setIsExecuting(true);
    setError(null);

    try {
      const result = await sqlAPI.execute(currentSql, activeDatabase);
      setQueryResult(result);
      
      // Add to history
      addToHistory({
        id: result.requestId,
        query: currentSql,
        sqlQuery: result.sqlQuery,
        timestamp: new Date().toISOString(),
        databaseId: activeDatabase,
        executionTime: result.executionTime,
        rowCount: result.rowCount,
      });

      toast.success(`SQL executed successfully! ${result.rowCount} rows returned`);
    } catch (error: any) {
      setError(error.response?.data?.error?.detail || 'Failed to execute SQL');
      toast.error('Failed to execute SQL');
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Query Interface</h1>
          <p className="text-gray-600">Ask questions in natural language or write SQL directly</p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={activeTab === 'natural' ? handleExecuteNaturalLanguage : handleExecuteSQL}
            disabled={isExecuting}
            className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isExecuting ? (
              <LoadingSpinner size="sm" className="mr-2" />
            ) : (
              <Play className="w-4 h-4 mr-2" />
            )}
            {isExecuting ? 'Executing...' : 'Execute'}
          </button>
          
          <button className="inline-flex items-center px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors">
            <Save className="w-4 h-4 mr-2" />
            Save
          </button>
        </div>
      </div>

      {/* Query Input */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as 'natural' | 'sql')}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="natural" className="flex items-center">
              <MessageSquare className="w-4 h-4 mr-2" />
              Natural Language
            </TabsTrigger>
            <TabsTrigger value="sql" className="flex items-center">
              <Code className="w-4 h-4 mr-2" />
              SQL Editor
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="natural" className="mt-0">
            <NaturalLanguageInput
              value={currentQuery}
              onChange={setCurrentQuery}
              onExecute={handleExecuteNaturalLanguage}
              isExecuting={isExecuting}
            />
          </TabsContent>
          
          <TabsContent value="sql" className="mt-0">
            <SqlEditor
              value={currentSql}
              onChange={setCurrentSql}
              onExecute={handleExecuteSQL}
              isExecuting={isExecuting}
            />
          </TabsContent>
        </Tabs>
      </div>

      {/* Results */}
      {queryResult && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <ResultsTable result={queryResult} />
        </div>
      )}

      {/* History Sidebar */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center">
            <History className="w-5 h-5 text-gray-500 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Query History</h3>
          </div>
        </div>
        <QueryHistory />
      </div>
    </div>
  );
};