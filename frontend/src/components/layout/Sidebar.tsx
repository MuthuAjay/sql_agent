import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Database,
  MessageSquare,
  Code2,
  BarChart3,
  History,
  Settings,
  Activity,
  Heart,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface SidebarProps {
  collapsed: boolean;
  onToggle: (collapsed: boolean) => void;
}

interface NavItemProps {
  to: string;
  icon: React.ReactNode;
  label: string;
  collapsed: boolean;
}

function NavItem({ to, icon, label, collapsed }: NavItemProps) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        cn(
          'flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors',
          'hover:bg-gray-100 dark:hover:bg-gray-800',
          isActive 
            ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400' 
            : 'text-gray-700 dark:text-gray-300',
          collapsed ? 'justify-center' : 'justify-start'
        )
      }
      title={collapsed ? label : undefined}
    >
      <span className="flex-shrink-0">{icon}</span>
      {!collapsed && <span className="ml-3">{label}</span>}
    </NavLink>
  );
}

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const navItems = [
    { to: '/', icon: <MessageSquare size={20} />, label: 'Query' },
    { to: '/sql', icon: <Code2 size={20} />, label: 'SQL Editor' },
    { to: '/schema', icon: <Database size={20} />, label: 'Schema' },
    { to: '/analytics', icon: <BarChart3 size={20} />, label: 'Analytics' },
    { to: '/history', icon: <History size={20} />, label: 'History' },
    { to: '/performance', icon: <Activity size={20} />, label: 'Performance' },
    { to: '/health', icon: <Heart size={20} />, label: 'Health' },
    { to: '/settings', icon: <Settings size={20} />, label: 'Settings' },
  ];

  return (
    <div
      className={cn(
        'bg-white dark:bg-gray-950 border-r border-gray-200 dark:border-gray-800 flex flex-col transition-all duration-300',
        collapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-800">
        {!collapsed && (
          <div className="flex items-center space-x-2">
            <Database className="h-8 w-8 text-blue-600" />
            <span className="font-bold text-lg text-gray-900 dark:text-white">SQL Agent</span>
          </div>
        )}
        <button
          onClick={() => onToggle(!collapsed)}
          className="p-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500 dark:text-gray-400"
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => (
          <NavItem key={item.to} {...item} collapsed={collapsed} />
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-800">
        <div className={cn(
          'text-xs text-gray-500 dark:text-gray-400',
          collapsed ? 'text-center' : 'text-left'
        )}>
          {collapsed ? 'v1.0' : 'SQL Agent v1.0'}
        </div>
      </div>
    </div>
  );
}