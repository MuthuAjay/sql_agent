import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  Database, 
  MessageSquare, 
  FileText, 
  BarChart3, 
  History, 
  Settings, 
  ChevronLeft,
  ChevronRight,
  Search
} from 'lucide-react';
import { useUIStore, useDatabaseStore } from '../../stores';
import { cn } from '../../utils/cn';

const navigation = [
  { name: 'Query', href: '/', icon: Search },
  { name: 'Schema', href: '/schema', icon: Database },
  { name: 'Chat', href: '/chat', icon: MessageSquare },
  { name: 'History', href: '/history', icon: History },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'Reports', href: '/reports', icon: FileText },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export const Sidebar: React.FC = () => {
  const { sidebarOpen, toggleSidebar } = useUIStore();
  const { databases } = useDatabaseStore();

  return (
    <div className={cn(
      'fixed left-0 top-16 h-full bg-white border-r border-gray-200 transition-all duration-300 z-20',
      sidebarOpen ? 'w-64' : 'w-16'
    )}>
      <div className="flex flex-col h-full">
        {/* Toggle Button */}
        <button
          onClick={toggleSidebar}
          className="flex items-center justify-center w-full h-12 border-b border-gray-200 hover:bg-gray-50 transition-colors"
        >
          {sidebarOpen ? (
            <ChevronLeft className="w-5 h-5 text-gray-500" />
          ) : (
            <ChevronRight className="w-5 h-5 text-gray-500" />
          )}
        </button>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) =>
                cn(
                  'flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                  isActive
                    ? 'bg-blue-50 text-blue-700'
                    : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                )
              }
            >
              <item.icon className="w-5 h-5 mr-3 flex-shrink-0" />
              {sidebarOpen && <span>{item.name}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Database Status */}
        {sidebarOpen && (
          <div className="p-4 border-t border-gray-200">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
              Databases
            </h3>
            <div className="space-y-2">
              {databases.length === 0 ? (
                <div className="text-sm text-red-500 text-center">
                  No databases available. Please check your backend connection.
                </div>
              ) : (
                databases.slice(0, 3).map((db) => (
                  <div key={db.id} className="flex items-center justify-between text-sm">
                    <span className="text-gray-700 truncate">{db.name}</span>
                    <div className={cn(
                      'w-2 h-2 rounded-full',
                      db.status === 'connected' ? 'bg-green-500' : 'bg-red-500'
                    )} />
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};