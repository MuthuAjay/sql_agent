import React, { useState } from "react";
import { Columns as ColumnsIcon, Search, Crown, Link as LinkIcon, ChevronDown } from "lucide-react";

interface Column {
  name: string;
  type: string;
  primaryKey?: boolean;
  foreignKey?: { referencedTable: string; referencedColumn: string } | null;
  nullable?: boolean;
  defaultValue?: string | null;
}

interface ColumnsCardProps {
  columns: Column[];
}

const ColumnsCard: React.FC<ColumnsCardProps> = ({ columns }) => {
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const filteredColumns = columns.filter((col) =>
    col.name.toLowerCase().includes(search.toLowerCase())
  );

  const toggleExpand = (name: string) => {
    setExpanded((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-sm">
      <div className="p-6 border-b border-slate-100">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-slate-800 flex items-center">
            <ColumnsIcon className="w-5 h-5 mr-2 text-blue-600" />
            Columns ({columns.length})
          </h3>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              className="pl-10 pr-4 py-2 border border-slate-300 rounded-lg text-sm"
              placeholder="Search columns..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
        </div>
      </div>
      <div className="p-4">
        <div className="space-y-3">
          {filteredColumns.length === 0 ? (
            <div className="text-slate-400 text-center py-8">No columns found</div>
          ) : (
            filteredColumns.map((column) => (
              <div
                key={column.name}
                className="group p-4 border border-slate-100 rounded-lg hover:border-blue-200 hover:shadow-sm transition-all"
              >
                <div className="flex items-center justify-between cursor-pointer" onClick={() => toggleExpand(column.name)}>
                  <div className="flex items-center space-x-3">
                    {column.primaryKey && <Crown className="w-4 h-4 text-yellow-500" />}
                    {column.foreignKey && <LinkIcon className="w-4 h-4 text-blue-500" />}
                    <span className="font-medium text-slate-800">{column.name}</span>
                    <span className="px-2 py-1 bg-slate-100 text-slate-600 text-xs rounded-md font-mono">
                      {column.type}
                    </span>
                    {!column.nullable && (
                      <span className="px-2 py-1 bg-red-100 text-red-700 text-xs rounded-md">
                        NOT NULL
                      </span>
                    )}
                  </div>
                  <ChevronDown className={`w-4 h-4 text-slate-400 group-hover:text-slate-600 transition-transform ${expanded[column.name] ? 'rotate-180' : ''}`} />
                </div>
                {/* Expandable details */}
                {expanded[column.name] && (
                  <div className="mt-3 text-sm text-slate-600">
                    {column.defaultValue && (
                      <div>Default: <code className="bg-slate-100 px-1 rounded">{column.defaultValue}</code></div>
                    )}
                    {column.foreignKey && (
                      <div>References: {column.foreignKey.referencedTable}.{column.foreignKey.referencedColumn}</div>
                    )}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default ColumnsCard; 