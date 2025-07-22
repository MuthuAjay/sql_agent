import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { BarChart3, TrendingUp, Users, Database, Calendar, Filter } from 'lucide-react';
import { api } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

// Mock data for demonstration - in a real app, this would come from your API
const mockAnalyticsData = {
  queryVolume: [
    { date: '2024-01-01', queries: 145, users: 23 },
    { date: '2024-01-02', queries: 167, users: 28 },
    { date: '2024-01-03', queries: 198, users: 31 },
    { date: '2024-01-04', queries: 213, users: 35 },
    { date: '2024-01-05', queries: 189, users: 29 },
    { date: '2024-01-06', queries: 234, users: 38 },
    { date: '2024-01-07', queries: 278, users: 42 },
  ],
  topTables: [
    { name: 'users', queries: 145, percentage: 32 },
    { name: 'orders', queries: 98, percentage: 22 },
    { name: 'products', queries: 87, percentage: 19 },
    { name: 'transactions', queries: 65, percentage: 14 },
    { name: 'analytics', queries: 58, percentage: 13 },
  ],
  queryTypes: [
    { name: 'SELECT', value: 68, color: '#3B82F6' },
    { name: 'INSERT', value: 15, color: '#10B981' },
    { name: 'UPDATE', value: 12, color: '#F59E0B' },
    { name: 'DELETE', value: 3, color: '#EF4444' },
    { name: 'CREATE', value: 2, color: '#8B5CF6' },
  ],
  performance: [
    { time: '00:00', avg_response: 120, queries: 15 },
    { time: '04:00', avg_response: 95, queries: 8 },
    { time: '08:00', avg_response: 180, queries: 45 },
    { time: '12:00', avg_response: 220, queries: 78 },
    { time: '16:00', avg_response: 195, queries: 65 },
    { time: '20:00', avg_response: 145, queries: 32 },
  ],
};

export function AnalyticsPage() {
  const [dateRange, setDateRange] = useState('7d');
  const [selectedMetric, setSelectedMetric] = useState('queries');

  const { data: performanceMetrics, isLoading } = useQuery({
    queryKey: ['performance-metrics', dateRange],
    queryFn: api.performance.getMetrics,
    refetchInterval: 30000,
  });

  const statsCards = [
    {
      title: 'Total Queries',
      value: '2,435',
      change: '+12.5%',
      trend: 'up',
      icon: Database,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50 dark:bg-blue-900/20',
    },
    {
      title: 'Active Users',
      value: '156',
      change: '+8.2%',
      trend: 'up',
      icon: Users,
      color: 'text-green-600',
      bgColor: 'bg-green-50 dark:bg-green-900/20',
    },
    {
      title: 'Avg Response Time',
      value: '165ms',
      change: '-5.3%',
      trend: 'down',
      icon: TrendingUp,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50 dark:bg-purple-900/20',
    },
    {
      title: 'Success Rate',
      value: '98.7%',
      change: '+0.3%',
      trend: 'up',
      icon: BarChart3,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50 dark:bg-orange-900/20',
    },
  ];

  if (isLoading) {
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
          <BarChart3 size={24} className="text-blue-600" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Analytics Dashboard</h1>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-2">
            <Calendar size={16} className="text-gray-500" />
            <select
              value={dateRange}
              onChange={(e) => setDateRange(e.target.value)}
              className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md px-3 py-2 text-sm"
            >
              <option value="1d">Last 24 hours</option>
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
            </select>
          </div>
          
          <Button variant="outline" size="sm">
            <Filter size={16} />
            Filters
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-6">
        {statsCards.map((stat, index) => {
          const IconComponent = stat.icon;
          return (
            <Card key={index}>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                      {stat.title}
                    </p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                      {stat.value}
                    </p>
                  </div>
                  <div className={`p-3 rounded-full ${stat.bgColor}`}>
                    <IconComponent size={24} className={stat.color} />
                  </div>
                </div>
                <div className="mt-4 flex items-center">
                  <span className={`text-sm font-medium ${
                    stat.trend === 'up' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                  }`}>
                    {stat.change}
                  </span>
                  <span className="text-sm text-gray-500 dark:text-gray-400 ml-1">
                    vs last period
                  </span>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
        {/* Query Volume Trend */}
        <Card>
          <CardHeader>
            <CardTitle>Query Volume Trend</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={mockAnalyticsData.queryVolume}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(156, 163, 175, 0.2)" />
                <XAxis dataKey="date" stroke="rgba(156, 163, 175, 0.8)" />
                <YAxis stroke="rgba(156, 163, 175, 0.8)" />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="queries"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={{ fill: '#3B82F6' }}
                />
                <Line
                  type="monotone"
                  dataKey="users"
                  stroke="#10B981"
                  strokeWidth={2}
                  dot={{ fill: '#10B981' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Query Types Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Query Types Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={mockAnalyticsData.queryTypes}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {mockAnalyticsData.queryTypes.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Top Queried Tables */}
        <Card>
          <CardHeader>
            <CardTitle>Most Queried Tables</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={mockAnalyticsData.topTables}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(156, 163, 175, 0.2)" />
                <XAxis dataKey="name" stroke="rgba(156, 163, 175, 0.8)" />
                <YAxis stroke="rgba(156, 163, 175, 0.8)" />
                <Tooltip />
                <Bar dataKey="queries" fill="#3B82F6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Performance Over Time */}
        <Card>
          <CardHeader>
            <CardTitle>Response Time & Load</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={mockAnalyticsData.performance}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(156, 163, 175, 0.2)" />
                <XAxis dataKey="time" stroke="rgba(156, 163, 175, 0.8)" />
                <YAxis stroke="rgba(156, 163, 175, 0.8)" />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="avg_response"
                  stroke="#EF4444"
                  strokeWidth={2}
                  name="Avg Response Time (ms)"
                />
                <Line
                  type="monotone"
                  dataKey="queries"
                  stroke="#10B981"
                  strokeWidth={2}
                  name="Query Count"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Tables */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Recent Query Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left py-2 font-medium text-gray-600 dark:text-gray-400">Query</th>
                    <th className="text-left py-2 font-medium text-gray-600 dark:text-gray-400">Duration</th>
                    <th className="text-left py-2 font-medium text-gray-600 dark:text-gray-400">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  <tr>
                    <td className="py-2 text-gray-900 dark:text-white">SELECT * FROM users WHERE...</td>
                    <td className="py-2 text-gray-600 dark:text-gray-400">125ms</td>
                    <td className="py-2">
                      <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 text-xs rounded">
                        Success
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2 text-gray-900 dark:text-white">SELECT COUNT(*) FROM orders...</td>
                    <td className="py-2 text-gray-600 dark:text-gray-400">89ms</td>
                    <td className="py-2">
                      <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 text-xs rounded">
                        Success
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2 text-gray-900 dark:text-white">UPDATE products SET...</td>
                    <td className="py-2 text-gray-600 dark:text-gray-400">234ms</td>
                    <td className="py-2">
                      <span className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300 text-xs rounded">
                        Error
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Top Users by Query Count</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { name: 'Alice Johnson', queries: 145, percentage: 89 },
                { name: 'Bob Smith', queries: 98, percentage: 60 },
                { name: 'Carol Davis', queries: 87, percentage: 53 },
                { name: 'David Wilson', queries: 65, percentage: 40 },
                { name: 'Eva Brown', queries: 58, percentage: 36 },
              ].map((user, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                      <span className="text-white font-medium text-sm">
                        {user.name.split(' ').map(n => n[0]).join('')}
                      </span>
                    </div>
                    <span className="font-medium text-gray-900 dark:text-white">{user.name}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">{user.queries}</span>
                    <div className="w-20 h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
                      <div
                        className="h-2 bg-blue-600 rounded-full transition-all duration-300"
                        style={{ width: `${user.percentage}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}