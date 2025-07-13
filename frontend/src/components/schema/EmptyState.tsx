import React from "react";
import { Database, Eye, Sparkles, Code } from "lucide-react";

const EmptyState: React.FC = () => {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center max-w-md">
        <div className="w-24 h-24 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-full flex items-center justify-center mx-auto mb-6">
          <Database className="w-12 h-12 text-blue-600" />
        </div>
        <h3 className="text-xl font-semibold text-slate-800 mb-2">Select a Table</h3>
        <p className="text-slate-600 mb-6">
          Choose a table from the sidebar to view its schema, sample data, and AI-generated insights.
        </p>
        <div className="flex items-center justify-center space-x-4 text-sm text-slate-500">
          <div className="flex items-center">
            <Eye className="w-4 h-4 mr-1" />
            View Schema
          </div>
          <div className="flex items-center">
            <Sparkles className="w-4 h-4 mr-1" />
            AI Insights
          </div>
          <div className="flex items-center">
            <Code className="w-4 h-4 mr-1" />
            Sample Data
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmptyState; 