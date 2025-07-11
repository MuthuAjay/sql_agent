import React from 'react';
import { Bell, User, Sun, Moon, Database } from 'lucide-react';
import { useUIStore, useDatabaseStore } from '../../stores';
import { cn } from '../../utils/cn';

export const TopNavigation: React.FC = () => {
  const { theme, setTheme, activeDatabase, setActiveDatabase } = useUIStore();
  const { databases } = useDatabaseStore();

  return (
    <header className="fixed top-0 left-0 right-0 bg-white border-b border-gray-200 z-30">
      <div className="flex items-center justify-between h-16 px-6">
        {/* Logo */}
        <div className="flex items-center">
          <Database className="w-8 h-8 text-blue-600 mr-3" />
          <h1 className="text-xl font-bold text-gray-900">SQL Agent</h1>
        </div>

        {/* Database Selector */}
        <div className="flex-1 max-w-md mx-8">
          <select
            value={activeDatabase || ''}
            onChange={(e) => setActiveDatabase(e.target.value || null)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">Select Database</option>
            {databases.map((db) => (
              <option key={db.id} value={db.id}>
                {db.name} ({db.type})
              </option>
            ))}
          </select>
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-4">
          {/* Theme Toggle */}
          <button
            onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
          >
            {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
          </button>

          {/* Notifications */}
          <button className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors">
            <Bell className="w-5 h-5" />
          </button>

          {/* User Profile */}
          <button className="flex items-center space-x-2 p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors">
            <User className="w-5 h-5" />
            <span className="hidden sm:inline text-sm font-medium">Profile</span>
          </button>
        </div>
      </div>
    </header>
  );
};