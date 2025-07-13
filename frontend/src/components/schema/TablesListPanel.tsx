import React, { useEffect, useState } from "react";
import { Search, Database, FileText } from "lucide-react";
import { useTableStore } from "../../stores";
import { useUIStore } from "../../stores";

const TablesListPanel: React.FC = () => {
  const [search, setSearch] = useState("");
  const activeDatabase = useUIStore((s) => s.activeDatabase);
  const tables = useTableStore((s) => s.tables) || [];
  const loading = useTableStore((s) => s.loadingTables);
  const fetchTables = useTableStore((s) => s.fetchTables);
  const selectedTable = useTableStore((s) => s.selectedTable);
  const setSelectedTable = useTableStore((s) => s.setSelectedTable);

  // Fetch tables when activeDatabase changes
  useEffect(() => {
    if (activeDatabase) {
      fetchTables(activeDatabase);
    }
  }, [activeDatabase, fetchTables]);

  const filteredTables = tables.filter((table) =>
    table.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full">
      {/* Header with search */}
      <div className="p-6 bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
        <h2 className="text-xl font-bold mb-4">Database Tables</h2>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-blue-300" />
          <input
            className="w-full pl-10 pr-4 py-2 bg-white/20 backdrop-blur border border-white/30 rounded-lg placeholder-blue-200 text-white"
            placeholder="Search tables..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            disabled={loading || !activeDatabase}
          />
        </div>
      </div>
      {/* Table items */}
      <div className="p-4 space-y-2 flex-1 overflow-y-auto">
        {!activeDatabase ? (
          <div className="flex flex-col items-center justify-center h-40 text-slate-400">
            <Database className="w-10 h-10 mb-2" />
            <div>Select a database to view tables</div>
          </div>
        ) : loading ? (
          <div className="space-y-2 animate-pulse">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-12 bg-slate-100/60 rounded-lg" />
            ))}
          </div>
        ) : filteredTables.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 text-slate-400">
            <Database className="w-10 h-10 mb-2" />
            <div>No tables found</div>
          </div>
        ) : (
          filteredTables.map((table) => (
            <div
              key={table.name}
              className={`group p-3 rounded-lg border border-slate-200 hover:border-blue-300 hover:shadow-md transition-all duration-200 cursor-pointer hover:scale-[1.02] ${selectedTable === table.name ? "border-blue-500 bg-blue-50" : ""}`}
              onClick={() => setSelectedTable(table.name)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {table.type === "table" ? (
                    <Database className="w-4 h-4 text-blue-500" />
                  ) : (
                    <FileText className="w-4 h-4 text-emerald-500" />
                  )}
                  <span className="font-medium text-slate-700">{table.name}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span
                    className={`px-2 py-1 text-xs rounded-full ${
                      table.type === "table"
                        ? "bg-blue-100 text-blue-700"
                        : "bg-emerald-100 text-emerald-700"
                    }`}
                  >
                    {table.type}
                  </span>
                  {table.rowCount ? (
                    <span className="text-xs text-slate-500">{table.rowCount} rows</span>
                  ) : null}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default TablesListPanel; 