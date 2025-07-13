import React from "react";
import { Database, Copy, Code } from "lucide-react";

interface TableHeaderCardProps {
  tableName: string;
  columnsCount: number;
  rowsCount: number;
}

const TableHeaderCard: React.FC<TableHeaderCardProps> = ({ tableName, columnsCount, rowsCount }) => {
  return (
    <div className="bg-gradient-to-r from-white to-blue-50 border border-slate-200 rounded-xl p-6 shadow-sm">
      <div className="flex items-start justify-between">
        <div className="flex items-center space-x-4">
          <div className="p-3 bg-blue-100 rounded-lg">
            <Database className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-800">{tableName}</h1>
            <div className="flex items-center space-x-4 mt-2">
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                <Database className="w-3 h-3 mr-1" />
                Table
              </span>
              <span className="text-slate-500">{columnsCount} columns • {rowsCount.toLocaleString()} rows</span>
            </div>
          </div>
        </div>
        <div className="flex space-x-2">
          <button className="px-4 py-2 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors flex items-center">
            <Copy className="w-4 h-4 mr-2" />
            Copy Name
          </button>
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center">
            <Code className="w-4 h-4 mr-2" />
            Generate SQL
          </button>
        </div>
      </div>
    </div>
  );
};

export default TableHeaderCard; 