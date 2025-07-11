import React from 'react';
import { FileText } from 'lucide-react';

export const ReportsPage: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center py-16">
        <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Reports</h2>
        <p className="text-gray-600">
          This page will show saved reports and report generation capabilities.
        </p>
      </div>
    </div>
  );
};