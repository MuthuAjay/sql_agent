import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { Wifi, WifiOff, Database, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

export function StatusBar() {
  const { data: health, isLoading } = useQuery({
    queryKey: ['health'],
    queryFn: api.health.check,
    refetchInterval: 30000, // Check every 30 seconds
    retry: false,
  });

  const apiStatus = health?.api_status || 'unknown';
  const dbStatus = health?.database_status || 'unknown';

  return (
    <div className="bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 px-4 py-2">
      <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400">
        <div className="flex items-center space-x-4">
          {/* API Status */}
          <div className="flex items-center space-x-1">
            {apiStatus === 'healthy' ? (
              <Wifi size={14} className="text-green-500" />
            ) : (
              <WifiOff size={14} className="text-red-500" />
            )}
            <span>API: {apiStatus}</span>
          </div>

          {/* Database Status */}
          <div className="flex items-center space-x-1">
            <Database 
              size={14} 
              className={cn(
                dbStatus === 'connected' ? 'text-green-500' : 'text-red-500'
              )} 
            />
            <span>DB: {dbStatus}</span>
          </div>

          {/* Response Time */}
          {health?.response_time && (
            <div className="flex items-center space-x-1">
              <span>Response: {health.response_time}ms</span>
            </div>
          )}
        </div>

        <div className="flex items-center space-x-4">
          {/* Error Count */}
          {health && health.error_count > 0 && (
            <div className="flex items-center space-x-1 text-orange-500">
              <AlertCircle size={14} />
              <span>{health.error_count} errors</span>
            </div>
          )}

          {/* Uptime */}
          {health?.uptime && (
            <span>Uptime: {Math.floor(health.uptime / 3600)}h</span>
          )}
        </div>
      </div>
    </div>
  );
}