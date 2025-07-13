import React from "react";
import { Link as LinkIcon } from "lucide-react";

interface Relationship {
  column: string;
  referencedTable: string;
  referencedColumn: string;
}

interface RelationshipsCardProps {
  relationships: Relationship[];
}

const RelationshipsCard: React.FC<RelationshipsCardProps> = ({ relationships }) => {
  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-sm">
      <div className="p-6 border-b border-slate-100">
        <h3 className="text-lg font-semibold text-slate-800 flex items-center">
          <LinkIcon className="w-5 h-5 mr-2 text-blue-600" />
          Relationships
        </h3>
      </div>
      <div className="p-4 space-y-3">
        {relationships.length === 0 ? (
          <div className="text-slate-400 text-center py-8">No relationships found</div>
        ) : (
          relationships.map((rel, i) => (
            <div key={i} className="p-4 border border-slate-100 rounded-lg flex items-center space-x-4">
              <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-700 font-mono">{rel.column}</span>
              <span className="text-xs text-slate-700">→ {rel.referencedTable}.{rel.referencedColumn}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default RelationshipsCard; 