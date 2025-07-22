import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Heart, Database, Wifi, AlertCircle, CheckCircle, Clock, Activity } from 'lucide-react';
import { api } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { formatDuration } from '@/lib/utils';

export function HealthPage() {
  const { data: health, isLoading, error } = useQuery({
    queryKey: ['health'],
    queryFn: api.health.check,
    refetchInterval: 5000, // Check every 5 seconds
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'connected':
        return <CheckCircle size={20} className="text-green-500" />;
      case 'degraded':
      case 'disconnected':
        return <AlertCircle size={20} className="text-yellow-500" />;
      case 'down':
      case 'error':
        return <AlertCircle size={20} className="text-red-500" />;
      default:
        return <Clock size={20} className="text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'connected':
        return 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
      case 'degraded':
      case 'disconnected':
        return 'text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
      case 'down':
      case 'error':
        return 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
      default:
        return 'text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full bg-gray-50 dark:bg-gray-900 p-6">
        <div className="flex items-center justify-center h-full">
          <Card className="max-w-md">
            <CardContent className="p-6 text-center">
              <AlertCircle size={48} className="text-red-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                Health Check Failed
              </h3>
              <p className="text-gray-600 dark:text-gray-400">
                Unable to retrieve system health information. The API might be down.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full bg-gray-50 dark:bg-gray-900 p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Heart size={24} className="text-red-600" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">System Health</h1>
        </div>
        
        <div className="text-sm text-gray-500 dark:text-gray-400">
          Last updated: {health?.last_check ? new Date(health.last_check).toLocaleString() : 'Never'}
        </div>
      </div>

      {/* Overall Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              {getStatusIcon(health?.api_status || 'unknown')}
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white">API Status</h3>
                <p className={`text-sm font-medium capitalize ${getStatusColor(health?.api_status || 'unknown')}`}>
                  {health?.api_status || 'Unknown'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              {getStatusIcon(health?.database_status || 'unknown')}
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white">Database</h3>
                <p className={`text-sm font-medium capitalize ${getStatusColor(health?.database_status || 'unknown')}`}>
                  {health?.database_status || 'Unknown'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <Activity size={20} className="text-blue-500" />
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white">Response Time</h3>
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {health?.response_time ? `${health.response_time}ms` : 'N/A'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Metrics */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
        <Card>
          <CardHeader>
            <CardTitle>System Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between py-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Uptime</span>
                <span className="text-sm text-gray-900 dark:text-white">
                  {health?.uptime ? formatDuration(health.uptime * 1000) : 'N/A'}
                </span>
              </div>
              
              <div className="flex items-center justify-between py-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Error Count</span>
                <span className={`text-sm font-medium ${
                  (health?.error_count || 0) > 10 
                    ? 'text-red-600 dark:text-red-400' 
                    : (health?.error_count || 0) > 5 
                    ? 'text-yellow-600 dark:text-yellow-400'
                    : 'text-green-600 dark:text-green-400'
                }`}>
                  {health?.error_count || 0}
                </span>
              </div>
              
              <div className="flex items-center justify-between py-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-sm font-medium text-gray-600 dark:text-gray-400">API Response Time</span>
                <span className={`text-sm font-medium ${
                  (health?.response_time || 0) > 1000 
                    ? 'text-red-600 dark:text-red-400' 
                    : (health?.response_time || 0) > 500 
                    ? 'text-yellow-600 dark:text-yellow-400'
                    : 'text-green-600 dark:text-green-400'
                }`}>
                  {health?.response_time ? `${health.response_time}ms` : 'N/A'}
                </span>
              </div>
              
              <div className="flex items-center justify-between py-2">
                <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Last Check</span>
                <span className="text-sm text-gray-900 dark:text-white">
                  {health?.last_check ? new Date(health.last_check).toLocaleString() : 'Never'}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Service Components */}
        <Card>
          <CardHeader>
            <CardTitle>Service Components</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
                <div className="flex items-center space-x-3">
                  <Wifi size={18} className={
                    health?.api_status === 'healthy' ? 'text-green-500' : 'text-red-500'
                  } />
                  <span className="font-medium text-gray-900 dark:text-white">API Server</span>
                </div>
                <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(health?.api_status || 'unknown')}`}>
                  {health?.api_status || 'Unknown'}
                </span>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
                <div className="flex items-center space-x-3">
                  <Database size={18} className={
                    health?.database_status === 'connected' ? 'text-green-500' : 'text-red-500'
                  } />
                  <span className="font-medium text-gray-900 dark:text-white">Database</span>
                </div>
                <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(health?.database_status || 'unknown')}`}>
                  {health?.database_status || 'Unknown'}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Health History/Alerts */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Health Events</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {health?.error_count && health.error_count > 0 ? (
              <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
                <div className="flex items-start space-x-3">
                  <AlertCircle size={20} className="text-red-500 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-red-800 dark:text-red-200">
                      High Error Count
                    </h4>
                    <p className="text-sm text-red-700 dark:text-red-300">
                      {health.error_count} errors detected in the system. Please check the logs for more details.
                    </p>
                    <p className="text-xs text-red-600 dark:text-red-400 mt-1">
                      {new Date().toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>
            ) : null}

            {health?.response_time && health.response_time > 500 ? (
              <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md">
                <div className="flex items-start space-x-3">
                  <Clock size={20} className="text-yellow-500 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-yellow-800 dark:text-yellow-200">
                      Slow Response Time
                    </h4>
                    <p className="text-sm text-yellow-700 dark:text-yellow-300">
                      API response time ({health.response_time}ms) is higher than normal.
                    </p>
                    <p className="text-xs text-yellow-600 dark:text-yellow-400 mt-1">
                      {new Date().toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>
            ) : null}

            {(!health?.error_count || health.error_count === 0) && 
             (!health?.response_time || health.response_time <= 500) && (
              <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md">
                <div className="flex items-start space-x-3">
                  <CheckCircle size={20} className="text-green-500 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-green-800 dark:text-green-200">
                      All Systems Operational
                    </h4>
                    <p className="text-sm text-green-700 dark:text-green-300">
                      All services are running normally with good performance.
                    </p>
                    <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                      {new Date().toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}