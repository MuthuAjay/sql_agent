import React, { useState, useRef, useEffect } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Send, Loader2, Lightbulb, AlertCircle, CheckCircle } from 'lucide-react';
import { api, APIError } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card, CardContent } from '@/components/ui/Card';
import { QueryResponse } from '@/types/api';
import { DataTable } from '@/components/data/DataTable';
import { ChartComponent } from '@/components/visualization/ChartComponent';
import { InsightsPanel } from '@/components/analysis/InsightsPanel';
import { debounce } from '@/lib/utils';
import toast from 'react-hot-toast';

interface QueryMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  response?: QueryResponse;
  error?: string;
}

export function QueryInterface() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<QueryMessage[]>([
    {
      id: 'welcome',
      type: 'assistant',
      content: 'Hello! I\'m your AI SQL assistant. Ask me anything about your data in natural language.',
      timestamp: new Date(),
    },
  ]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const queryMutation = useMutation({
    mutationFn: api.query.process,
    onSuccess: (response) => {
      const assistantMessage: QueryMessage = {
        id: `assistant-${Date.now()}`,
        type: 'assistant',
        content: `I've executed your query successfully. Here are the results:`,
        timestamp: new Date(),
        response,
      };
      setMessages(prev => [...prev, assistantMessage]);
      toast.success('Query executed successfully!');
    },
    onError: (error: APIError) => {
      const errorMessage: QueryMessage = {
        id: `error-${Date.now()}`,
        type: 'assistant',
        content: 'Sorry, I encountered an error while processing your query.',
        timestamp: new Date(),
        error: error.message,
      };
      setMessages(prev => [...prev, errorMessage]);
      toast.error(error.message);
    },
  });

  const getSuggestions = useMutation({
    mutationFn: api.query.getSuggestions,
    onSuccess: (suggestions) => {
      setSuggestions(suggestions);
    },
  });

  const debouncedGetSuggestions = debounce((query: string) => {
    if (query.length > 3) {
      getSuggestions.mutate(query);
    } else {
      setSuggestions([]);
    }
  }, 300);

  useEffect(() => {
    debouncedGetSuggestions(query);
  }, [query]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || queryMutation.isPending) return;

    const userMessage: QueryMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: query,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    queryMutation.mutate(query, {
      includeAnalysis: true,
      includeVisualization: true,
    });
    setQuery('');
    setSuggestions([]);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    setSuggestions([]);
  };

  const formatTimestamp = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="h-full flex flex-col">
      {/* Messages */}
      <div className="flex-1 overflow-auto p-6 space-y-4">
        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] ${message.type === 'user' ? 'order-2' : 'order-1'}`}>
              <Card className={message.type === 'user' ? 'bg-blue-50 dark:bg-blue-900/20' : ''}>
                <CardContent className="p-4">
                  <div className="flex items-start space-x-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      message.type === 'user' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                    }`}>
                      {message.type === 'user' ? 'U' : 'AI'}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-gray-900 dark:text-gray-100 mb-1">{message.content}</p>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {formatTimestamp(message.timestamp)}
                      </span>
                      
                      {message.error && (
                        <div className="mt-2 p-2 bg-red-50 dark:bg-red-900/20 rounded-md border border-red-200 dark:border-red-800">
                          <div className="flex items-start space-x-2">
                            <AlertCircle size={16} className="text-red-500 mt-0.5" />
                            <span className="text-sm text-red-700 dark:text-red-300">{message.error}</span>
                          </div>
                        </div>
                      )}

                      {message.response && (
                        <div className="mt-4 space-y-4">
                          {/* SQL and Execution Info */}
                          {message.response.sql_result && (
                            <div>
                              <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                                <div className="flex items-center justify-between mb-2">
                                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Generated SQL:</span>
                                  <div className="flex items-center space-x-2 text-xs text-gray-500 dark:text-gray-400">
                                    <CheckCircle size={14} className="text-green-500" />
                                    <span>{message.response.sql_result.execution_time}ms</span>
                                  </div>
                                </div>
                                <code className="text-sm font-mono bg-white dark:bg-gray-900 p-2 rounded border block">
                                  {message.response.sql_result.sql}
                                </code>
                              </div>

                              {/* Data Table */}
                              {message.response.sql_result.data.length > 0 && (
                                <DataTable
                                  data={message.response.sql_result.data}
                                  columns={message.response.sql_result.columns}
                                  totalRows={message.response.sql_result.total_rows}
                                />
                              )}
                            </div>
                          )}

                          {/* Visualization */}
                          {message.response.visualization_result && (
                            <ChartComponent visualization={message.response.visualization_result} />
                          )}

                          {/* Analysis Insights */}
                          {message.response.analysis_result && (
                            <InsightsPanel analysis={message.response.analysis_result} />
                          )}

                          {/* Suggestions */}
                          {message.response.suggestions.length > 0 && (
                            <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-md">
                              <div className="flex items-center space-x-2 mb-2">
                                <Lightbulb size={16} className="text-blue-500" />
                                <span className="text-sm font-medium text-blue-700 dark:text-blue-300">Suggestions:</span>
                              </div>
                              <div className="space-y-1">
                                {message.response.suggestions.map((suggestion, index) => (
                                  <button
                                    key={index}
                                    onClick={() => handleSuggestionClick(suggestion)}
                                    className="block w-full text-left text-sm text-blue-600 dark:text-blue-400 hover:underline"
                                  >
                                    • {suggestion}
                                  </button>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Query Input */}
      <div className="border-t border-gray-200 dark:border-gray-800 p-4">
        <form onSubmit={handleSubmit} className="space-y-2">
          {/* Suggestions */}
          {suggestions.length > 0 && (
            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg">
              <div className="p-2 space-y-1">
                {suggestions.slice(0, 5).map((suggestion, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="block w-full text-left px-2 py-1 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 rounded"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="flex space-x-2">
            <div className="flex-1">
              <Input
                type="text"
                placeholder="Ask me anything about your data..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={queryMutation.isPending}
                className="w-full"
              />
            </div>
            <Button
              type="submit"
              disabled={!query.trim() || queryMutation.isPending}
              loading={queryMutation.isPending}
              className="px-6"
            >
              {queryMutation.isPending ? <Loader2 size={16} /> : <Send size={16} />}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}