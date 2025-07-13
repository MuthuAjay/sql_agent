import React from "react";
import { Eye } from "lucide-react";

interface SampleData {
  columns: string[];
  rows: (string | number | null)[][];
}

interface SampleDataCardProps {
  data: SampleData | null;
}

const SampleDataCard: React.FC<SampleDataCardProps> = ({ data }) => {
  if (!data || !data.columns || data.columns.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 text-slate-400 text-center">
        No sample data available
      </div>
    );
  }
  const { columns, rows } = data;
  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-sm">
      <div className="p-6 border-b border-slate-100">
        <h3 className="text-lg font-semibold text-slate-800 flex items-center">
          <Eye className="w-5 h-5 mr-2 text-green-600" />
          Sample Data
        </h3>
      </div>
      <div className="p-4">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                {columns.map((col) => (
                  <th key={col} className="text-left p-3 font-medium text-slate-700 bg-slate-50">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i} className="border-b border-slate-100 hover:bg-slate-50">
                  {row.map((cell, j) => (
                    <td key={j} className="p-3 text-slate-600 font-mono text-xs">{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default SampleDataCard; 