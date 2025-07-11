import React, { useEffect } from 'react';
import { Clock, Database, BarChart, Copy } from 'lucide-react';
import { useQueryStore } from '../../stores';
import { queryAPI } from '../../services/api';
import { formatDistanceToNow } from 'date-fns';
import { LoadingSpinner } from '../common/LoadingSpinner';
import toast from 'react-hot-toast';

export const QueryHistory: React.FC = () => {
  const { queryHistory, setQueryHistory, setCurrentQuery, setCurrentSql } = useQueryStore();
  const [loading, setLoading] = React.useState(true);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const result = await queryAPI.getHistory();
        setQueryHistory(result.queries);
      } catch (error) {
        console.error('Failed to load query history:', error);
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, [setQueryHistory]);

  const handleCopyQuery = (query: string) => {
    navigator.clipboard.writeText(query);
    toast.success('Query copied to clipboard');
  };

  const handleUseQuery = (query: string, sql: string) => {
    setCurrentQuery(query);
    setCurrentSql(sql);
    toast.success('Query loaded to editor');
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <div className="max-h-96 overflow-y-auto">
      {queryHistory.length === 0 ? (
        <div className="text-center py-8">
          <Clock className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500">No queries yet</p>
          <p className="text-sm text-gray-400">Your recent queries will appear here</p>
        </div>
      ) : (
        <div className="divide-y divide-gray-200">
          {queryHistory.map((query) => (
            <div key={query.id} className="p-4 hover:bg-gray-50 transition-colors">
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {query.query}
                  </p>
                  <p className="text-xs text-gray-500 mt-1 font-mono">
                    {query.sqlQuery}
                  </p>
                  <div className="flex items-center mt-2 text-xs text-gray-500 space-x-4">
                    <div className="flex items-center">
                      <Database className="w-3 h-3 mr-1" />
                      {query.databaseId}
                    </div>
                    <div className="flex items-center">
                      <BarChart className="w-3 h-3 mr-1" />
                      {query.rowCount} rows
                    </div>
                    <div className="flex items-center">
                      <Clock className="w-3 h-3 mr-1" />
                      {formatDistanceToNow(new Date(query.timestamp), { addSuffix: true })}
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2 ml-4">
                  <button
                    onClick={() => handleCopyQuery(query.query)}
                    className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
                    title="Copy query"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleUseQuery(query.query, query.sqlQuery)}
                    className="px-2 py-1 text-xs text-blue-600 hover:bg-blue-50 rounded transition-colors"
                  >
                    Use
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};