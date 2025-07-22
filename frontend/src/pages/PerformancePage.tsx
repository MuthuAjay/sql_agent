import React, { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Activity, Play, Clock, TrendingUp, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';
import { api } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { formatDuration, formatNumber } from '@/lib/utils';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import toast from 'react-hot-toast';

// Mock performance data for demonstration
const mockPerformanceData = {
  responseTime: [
    { time: '00:00', avg: 120, p95: 200, p99: 350 },
    { time: '02:00', avg: 95, p95: 160, p99: 280 },
    { time: '04:00', avg: 88, p95: 145, p99: 260 },
    { time: '06:00', avg: 110, p95: 180, p99: 320 },
    { time: '08:00', avg: 180, p95: 280, p99: 450 },
    { time: '10:00', avg: 220, p95: 350, p99: 580 },
    { time: '12:00', avg: 195, p95: 320, p99: 520 },
    { time: '14:00', avg: 210, p95: 340, p99: 560 },
    { time: '16:00', avg: 185, p95: 290, p99: 480 },
    { time: '18:00', avg: 165, p95: 260, p99: 420 },
    { time: '20:00', avg: 145, p95: 230, p99: 380 },
    { time: '22:00', avg: 125, p95: 190, p99: 310 },
  ],
  throughput: [
    { time: '00:00', queries: 15, errors: 0 },
    { time: '02:00', queries: 8, errors: 0 },
    { time: '04:00', queries: 5, errors: 0 },
    { time: '06:00', queries: 12, errors: 1 },
    { time: '08:00', queries: 45, errors: 2 },
    { time: '10:00', queries: 78, errors: 3 },
    { time: '12:00', queries: 65, errors: 1 },
    { time: '14:00', queries: 72, errors: 2 },
    { time: '16:00', queries: 58, errors: 1 },
    { time: '18:00', queries: 42, errors: 0 },
    { time: '20:00', queries: 32, errors: 1 },
    { time: '22:00', queries: 18, errors: 0 },
  ],
};

export function PerformancePage() {
  const [testRunning, setTestRunning] = useState(false);
  const [testResults, setTestResults] = useState<any>(null);
  const [numQueries, setNumQueries] = useState(10);

  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['performance-metrics'],
    queryFn: api.performance.getMetrics,
    refetchInterval: 5000,
  });

  const performanceTest = useMutation({
    mutationFn: (numQueries: number) => api.performance.test(numQueries),
    onMutate: () => {
      setTestRunning(true);
    },
    onSuccess: (result) => {
      setTestResults(result);
      setTestRunning(false);
      toast.success('Performance test completed successfully');
    },
    onError: (error) => {
      setTestRunning(false);
      toast.error(`Performance test failed: ${error.message}`);
    },
  });

  const runPerformanceTest = () => {
    if (numQueries < 1 || numQueries > 1000) {
      toast.error('Number of queries must be between 1 and 1000');
      return;
    }
    performanceTest.mutate(numQueries);
  };

  const formatResponseTime = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const getHealthColor = (value: number, thresholds: { good: number; warning: number }) => {
    if (value <= thresholds.good) return 'text-green-600 dark:text-green-400';
    if (value <= thresholds.warning) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  if (metricsLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="h-full bg-gray-50 dark:bg-gray-900 p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Activity size={24} className="text-blue-600" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Performance Monitoring</h1>
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Avg Response Time
                </p>
                <p className={`text-2xl font-bold ${getHealthColor(metrics?.avg_response_time || 165, { good: 200, warning: 500 })}`}>
                  {formatResponseTime(metrics?.avg_response_time || 165)}
                </p>
              </div>
              <div className="p-3 rounded-full bg-blue-50 dark:bg-blue-900/20">
                <Clock size={24} className="text-blue-600" />
              </div>
            </div>
            <div className="mt-4 flex items-center">
              <span className="text-sm text-green-600 dark:text-green-400 font-medium">
                -5.3%
              </span>
              <span className="text-sm text-gray-500 dark:text-gray-400 ml-1">
                vs last hour
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Total Queries
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {formatNumber(metrics?.total_queries || 2435)}
                </p>
              </div>
              <div className="p-3 rounded-full bg-green-50 dark:bg-green-900/20">
                <TrendingUp size={24} className="text-green-600" />
              </div>
            </div>
            <div className="mt-4 flex items-center">
              <span className="text-sm text-green-600 dark:text-green-400 font-medium">
                +12.5%
              </span>
              <span className="text-sm text-gray-500 dark:text-gray-400 ml-1">
                vs yesterday
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Error Rate
                </p>
                <p className={`text-2xl font-bold ${getHealthColor((metrics?.error_rate || 0.013) * 100, { good: 1, warning: 5 })}`}>
                  {((metrics?.error_rate || 0.013) * 100).toFixed(2)}%
                </p>
              </div>
              <div className="p-3 rounded-full bg-red-50 dark:bg-red-900/20">
                <AlertCircle size={24} className="text-red-600" />
              </div>
            </div>
            <div className="mt-4 flex items-center">
              <span className="text-sm text-green-600 dark:text-green-400 font-medium">
                -0.3%
              </span>
              <span className="text-sm text-gray-500 dark:text-gray-400 ml-1">
                vs last hour
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Uptime
                </p>
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {((metrics?.uptime || 86400) / 3600).toFixed(1)}h
                </p>
              </div>
              <div className="p-3 rounded-full bg-green-50 dark:bg-green-900/20">
                <CheckCircle size={24} className="text-green-600" />
              </div>
            </div>
            <div className="mt-4 flex items-center">
              <span className="text-sm text-green-600 dark:text-green-400 font-medium">
                99.9% SLA
              </span>
              <span className="text-sm text-gray-500 dark:text-gray-400 ml-1">
                this month
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
        {/* Response Time Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Response Time Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={mockPerformanceData.responseTime}>
                <defs>
                  <linearGradient id="colorAvg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.1}/>
                  </linearGradient>
                  <linearGradient id="colorP95" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#F59E0B" stopOpacity={0.1}/>
                  </linearGradient>
                  <linearGradient id="colorP99" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#EF4444" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#EF4444" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(156, 163, 175, 0.2)" />
                <XAxis dataKey="time" stroke="rgba(156, 163, 175, 0.8)" />
                <YAxis stroke="rgba(156, 163, 175, 0.8)" />
                <Tooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="avg"
                  stroke="#3B82F6"
                  fillOpacity={1}
                  fill="url(#colorAvg)"
                  name="Average"
                />
                <Area
                  type="monotone"
                  dataKey="p95"
                  stroke="#F59E0B"
                  fillOpacity={1}
                  fill="url(#colorP95)"
                  name="95th Percentile"
                />
                <Area
                  type="monotone"
                  dataKey="p99"
                  stroke="#EF4444"
                  fillOpacity={1}
                  fill="url(#colorP99)"
                  name="99th Percentile"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Throughput Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Query Throughput & Errors</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={mockPerformanceData.throughput}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(156, 163, 175, 0.2)" />
                <XAxis dataKey="time" stroke="rgba(156, 163, 175, 0.8)" />
                <YAxis stroke="rgba(156, 163, 175, 0.8)" />
                <Tooltip />
                <Legend />
                <Bar dataKey="queries" fill="#10B981" name="Successful Queries" />
                <Bar dataKey="errors" fill="#EF4444" name="Errors" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Performance Testing */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Performance Testing</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4 mb-6">
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Number of queries:
              </label>
              <Input
                type="number"
                value={numQueries}
                onChange={(e) => setNumQueries(parseInt(e.target.value) || 10)}
                min="1"
                max="1000"
                className="w-24"
                disabled={testRunning}
              />
            </div>
            
            <Button
              onClick={runPerformanceTest}
              disabled={testRunning}
              loading={testRunning}
            >
              {testRunning ? <Loader2 size={16} /> : <Play size={16} />}
              {testRunning ? 'Running Test...' : 'Run Test'}
            </Button>
          </div>

          {testResults && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-md">
                <p className="text-sm font-medium text-blue-800 dark:text-blue-200">Total Time</p>
                <p className="text-xl font-bold text-blue-900 dark:text-blue-100">
                  {formatDuration(testResults.total_time)}
                </p>
              </div>
              
              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-md">
                <p className="text-sm font-medium text-green-800 dark:text-green-200">Successful</p>
                <p className="text-xl font-bold text-green-900 dark:text-green-100">
                  {testResults.successful}
                </p>
              </div>
              
              <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-md">
                <p className="text-sm font-medium text-red-800 dark:text-red-200">Failed</p>
                <p className="text-xl font-bold text-red-900 dark:text-red-100">
                  {testResults.failed}
                </p>
              </div>
              
              <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-md">
                <p className="text-sm font-medium text-purple-800 dark:text-purple-200">QPS</p>
                <p className="text-xl font-bold text-purple-900 dark:text-purple-100">
                  {testResults.queries_per_second.toFixed(2)}
                </p>
              </div>
            </div>
          )}

          {testResults && testResults.error_details && testResults.error_details.length > 0 && (
            <div className="mt-4 bg-red-50 dark:bg-red-900/20 p-4 rounded-md">
              <p className="font-medium text-red-800 dark:text-red-200 mb-2">Error Details:</p>
              <ul className="text-sm text-red-700 dark:text-red-300 space-y-1">
                {testResults.error_details.map((error: string, index: number) => (
                  <li key={index}>• {error}</li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Recent Performance Issues */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Performance Issues</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[
              {
                severity: 'high',
                type: 'Slow Query',
                message: 'Query on users table taking >2s',
                timestamp: '2024-01-15T10:30:00Z',
                resolved: false,
              },
              {
                severity: 'medium',
                type: 'High Load',
                message: 'API response time increased by 40%',
                timestamp: '2024-01-15T09:45:00Z',
                resolved: true,
              },
              {
                severity: 'low',
                type: 'Connection Pool',
                message: 'Database connections reaching 80% capacity',
                timestamp: '2024-01-15T08:15:00Z',
                resolved: true,
              },
            ].map((issue, index) => (
              <div
                key={index}
                className={`p-4 rounded-md border-l-4 ${
                  issue.severity === 'high'
                    ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                    : issue.severity === 'medium'
                    ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                    : 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center space-x-2 mb-1">
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded-full ${
                          issue.severity === 'high'
                            ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                            : issue.severity === 'medium'
                            ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
                            : 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'
                        }`}
                      >
                        {issue.severity}
                      </span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {issue.type}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                      {issue.message}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {new Date(issue.timestamp).toLocaleString()}
                    </p>
                  </div>
                  <span
                    className={`px-2 py-1 text-xs font-medium rounded-full ${
                      issue.resolved
                        ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                        : 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300'
                    }`}
                  >
                    {issue.resolved ? 'Resolved' : 'Active'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}