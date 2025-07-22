import React, { useState } from 'react';
import { History, Search, Calendar, Star, StarOff, Download, Play, Trash2, Filter } from 'lucide-react';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { formatDuration } from '@/lib/utils';
import { cn } from '@/lib/utils';
import toast from 'react-hot-toast';

interface QueryHistoryItem {
  id: string;
  query: string;
  sql?: string;
  timestamp: Date;
  duration: number;
  status: 'success' | 'error' | 'running';
  results?: {
    rows: number;
    columns: string[];
  };
  favorited: boolean;
  type: 'natural' | 'sql';
}

// Mock data for demonstration
const mockHistory: QueryHistoryItem[] = [
  {
    id: '1',
    query: 'Show me all users who signed up this month',
    sql: 'SELECT * FROM users WHERE created_at >= DATE_TRUNC(\'month\', CURRENT_DATE)',
    timestamp: new Date('2024-01-15T10:30:00'),
    duration: 125,
    status: 'success',
    results: { rows: 45, columns: ['id', 'name', 'email', 'created_at'] },
    favorited: true,
    type: 'natural',
  },
  {
    id: '2',
    query: 'SELECT COUNT(*) FROM orders WHERE status = \'completed\'',
    timestamp: new Date('2024-01-15T09:45:00'),
    duration: 89,
    status: 'success',
    results: { rows: 1, columns: ['count'] },
    favorited: false,
    type: 'sql',
  },
  {
    id: '3',
    query: 'What are the top selling products?',
    sql: 'SELECT p.name, SUM(oi.quantity) as total_sold FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.name ORDER BY total_sold DESC LIMIT 10',
    timestamp: new Date('2024-01-14T16:20:00'),
    duration: 234,
    status: 'success',
    results: { rows: 10, columns: ['name', 'total_sold'] },
    favorited: true,
    type: 'natural',
  },
  {
    id: '4',
    query: 'UPDATE products SET price = price * 1.1',
    timestamp: new Date('2024-01-14T14:15:00'),
    duration: 456,
    status: 'error',
    favorited: false,
    type: 'sql',
  },
  {
    id: '5',
    query: 'Show revenue by month for the last year',
    sql: 'SELECT DATE_TRUNC(\'month\', created_at) as month, SUM(total) as revenue FROM orders WHERE created_at >= NOW() - INTERVAL \'1 year\' GROUP BY month ORDER BY month',
    timestamp: new Date('2024-01-13T11:30:00'),
    duration: 178,
    status: 'success',
    results: { rows: 12, columns: ['month', 'revenue'] },
    favorited: false,
    type: 'natural',
  },
];

export function HistoryPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'natural' | 'sql' | 'favorites'>('all');
  const [filterStatus, setFilterStatus] = useState<'all' | 'success' | 'error'>('all');
  const [history, setHistory] = useLocalStorage<QueryHistoryItem[]>('query-history', mockHistory);
  
  const filteredHistory = history
    .filter(item => {
      if (searchQuery && !item.query.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }
      if (filterType === 'natural' && item.type !== 'natural') return false;
      if (filterType === 'sql' && item.type !== 'sql') return false;
      if (filterType === 'favorites' && !item.favorited) return false;
      if (filterStatus === 'success' && item.status !== 'success') return false;
      if (filterStatus === 'error' && item.status !== 'error') return false;
      return true;
    })
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

  const toggleFavorite = (id: string) => {
    setHistory(prev => prev.map(item => 
      item.id === id ? { ...item, favorited: !item.favorited } : item
    ));
    toast.success('Query favorite status updated');
  };

  const deleteQuery = (id: string) => {
    if (confirm('Are you sure you want to delete this query from history?')) {
      setHistory(prev => prev.filter(item => item.id !== id));
      toast.success('Query deleted from history');
    }
  };

  const executeQuery = (item: QueryHistoryItem) => {
    toast.info(`Would execute: ${item.query}`);
    // In a real app, this would execute the query again
  };

  const exportHistory = () => {
    const csv = [
      'Query,Type,Status,Duration,Results,Timestamp',
      ...filteredHistory.map(item => [
        `"${item.query}"`,
        item.type,
        item.status,
        item.duration,
        item.results ? item.results.rows : 0,
        item.timestamp.toISOString()
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `query-history-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20';
      case 'error': return 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20';
      case 'running': return 'text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20';
      default: return 'text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-900/20';
    }
  };

  const getTypeColor = (type: string) => {
    return type === 'natural' 
      ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20'
      : 'text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-900/20';
  };

  return (
    <div className="h-full bg-gray-50 dark:bg-gray-900 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <History size={24} className="text-blue-600" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Query History</h1>
        </div>
        
        <Button variant="outline" onClick={exportHistory}>
          <Download size={16} />
          Export History
        </Button>
      </div>

      {/* Filters */}
      <Card className="mb-6">
        <CardContent className="p-4">
          <div className="flex flex-col lg:flex-row lg:items-center space-y-4 lg:space-y-0 lg:space-x-4">
            {/* Search */}
            <div className="flex-1 relative">
              <Search size={18} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <Input
                type="text"
                placeholder="Search queries..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>

            {/* Type Filter */}
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as any)}
              className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md px-3 py-2 text-sm"
            >
              <option value="all">All Types</option>
              <option value="natural">Natural Language</option>
              <option value="sql">Direct SQL</option>
              <option value="favorites">Favorites Only</option>
            </select>

            {/* Status Filter */}
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value as any)}
              className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md px-3 py-2 text-sm"
            >
              <option value="all">All Status</option>
              <option value="success">Success</option>
              <option value="error">Error</option>
            </select>
          </div>
        </CardContent>
      </Card>

      {/* Query List */}
      <div className="space-y-4 max-h-[calc(100vh-280px)] overflow-auto">
        {filteredHistory.length > 0 ? (
          filteredHistory.map((item) => (
            <Card key={item.id} className="hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    {/* Query Text */}
                    <div className="flex items-start space-x-3 mb-3">
                      <div className="flex-1">
                        <p className="text-sm font-medium text-gray-900 dark:text-white mb-2 leading-relaxed">
                          {item.query}
                        </p>
                        
                        {/* Generated SQL for natural language queries */}
                        {item.sql && item.type === 'natural' && (
                          <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                            <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
                              Generated SQL:
                            </p>
                            <code className="text-xs text-gray-700 dark:text-gray-300 font-mono">
                              {item.sql}
                            </code>
                          </div>
                        )}
                      </div>
                      
                      {/* Actions */}
                      <div className="flex items-center space-x-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleFavorite(item.id)}
                          className="text-gray-400 hover:text-yellow-500"
                        >
                          {item.favorited ? <Star size={16} fill="currentColor" /> : <StarOff size={16} />}
                        </Button>
                        
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => executeQuery(item)}
                          className="text-gray-400 hover:text-blue-500"
                        >
                          <Play size={16} />
                        </Button>
                        
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => deleteQuery(item.id)}
                          className="text-gray-400 hover:text-red-500"
                        >
                          <Trash2 size={16} />
                        </Button>
                      </div>
                    </div>

                    {/* Metadata */}
                    <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                      <div className="flex items-center space-x-1">
                        <Calendar size={12} />
                        <span>{new Date(item.timestamp).toLocaleString()}</span>
                      </div>
                      
                      <span className={cn('px-2 py-1 rounded-full font-medium', getStatusColor(item.status))}>
                        {item.status}
                      </span>
                      
                      <span className={cn('px-2 py-1 rounded-full font-medium', getTypeColor(item.type))}>
                        {item.type === 'natural' ? 'Natural Language' : 'SQL'}
                      </span>
                      
                      <span>Duration: {formatDuration(item.duration)}</span>
                      
                      {item.results && (
                        <span>Results: {item.results.rows} rows</span>
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))
        ) : (
          <div className="text-center py-12">
            <History size={48} className="mx-auto text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              No queries found
            </h3>
            <p className="text-gray-500 dark:text-gray-400">
              {searchQuery || filterType !== 'all' || filterStatus !== 'all'
                ? 'Try adjusting your filters'
                : 'Your query history will appear here'
              }
            </p>
          </div>
        )}
      </div>
    </div>
  );
}