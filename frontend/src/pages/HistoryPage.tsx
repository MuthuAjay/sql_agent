import React from 'react';
import { History } from 'lucide-react';

export const HistoryPage: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center py-16">
        <History className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Query History</h2>
        <p className="text-gray-600">
          This page will show your complete query history with filtering and search capabilities.
        </p>
      </div>
    </div>
  );
};