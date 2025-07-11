import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { TopNavigation } from './TopNavigation';
import { useUIStore } from '../../stores';
import { cn } from '../../utils/cn';

export const AppLayout: React.FC = () => {
  const { sidebarOpen, theme } = useUIStore();

  return (
    <div className={cn('min-h-screen bg-gray-50', theme === 'dark' && 'dark bg-gray-900')}>
      <TopNavigation />
      <div className="flex">
        <Sidebar />
        <main
          className={cn(
            'flex-1 transition-all duration-300 pt-16',
            sidebarOpen ? 'ml-64' : 'ml-16'
          )}
        >
          <div className="p-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};