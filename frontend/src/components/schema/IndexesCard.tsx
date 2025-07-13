import React from "react";

interface Index {
  name: string;
  columns: string[];
  unique: boolean;
}

interface IndexesCardProps {
  indexes: Index[];
}

const IndexesCard: React.FC<IndexesCardProps> = ({ indexes }) => {
  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-sm">
      <div className="p-6 border-b border-slate-100">
        <h3 className="text-lg font-semibold text-slate-800">Indexes</h3>
      </div>
      <div className="p-4 space-y-3">
        {indexes.length === 0 ? (
          <div className="text-slate-400 text-center py-8">No indexes found</div>
        ) : (
          indexes.map((idx) => (
            <div key={idx.name} className="p-4 border border-slate-100 rounded-lg flex items-center space-x-4">
              <span className={`px-2 py-1 text-xs rounded-full font-mono ${idx.unique ? 'bg-blue-100 text-blue-700' : 'bg-slate-100 text-slate-600'}`}>{idx.name}</span>
              <span className="text-xs text-slate-500">{idx.unique ? "UNIQUE" : "NON-UNIQUE"}</span>
              <span className="text-xs text-slate-700">Columns: {idx.columns.join(", ")}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default IndexesCard; 