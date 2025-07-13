import React from "react";
import { Sparkles, RefreshCw, Brain } from "lucide-react";
import { useTableStore, useUIStore } from "../../stores";

function timeAgo(date: Date): string {
  const now = new Date();
  const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

interface DescriptionCardProps {
  tableName: string;
}

const DescriptionCard: React.FC<DescriptionCardProps> = ({ tableName }) => {
  const loading = useTableStore((s) => s.loadingDescription);
  const tableSchema = useTableStore((s) => s.tableSchema);
  const activeDatabase = useUIStore((s) => s.activeDatabase);
  const generateTableDescription = useTableStore((s) => s.generateTableDescription);

  const description = tableSchema?.description || null;
  const generatedAt = tableSchema?.descriptionGeneratedAt ? new Date(tableSchema.descriptionGeneratedAt) : null;

  const handleRegenerate = () => {
    if (activeDatabase && tableName) {
      generateTableDescription(activeDatabase, tableName, true);
    }
  };

  return (
    <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-xl shadow-sm">
      <div className="p-6 border-b border-purple-100">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-slate-800 flex items-center">
            <Sparkles className="w-5 h-5 mr-2 text-purple-600" />
            AI Description
          </h3>
          <button
            className="px-3 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm flex items-center"
            onClick={handleRegenerate}
            disabled={loading}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            {loading ? "Regenerating..." : "Regenerate"}
          </button>
        </div>
      </div>
      <div className="p-6">
        {description ? (
          <div className="space-y-4">
            <p className="text-slate-700 leading-relaxed">{description}</p>
            <div className="text-xs text-slate-500">
              Generated {generatedAt ? timeAgo(generatedAt) : "recently"} • Click regenerate for a fresh perspective
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <Brain className="w-12 h-12 text-purple-400 mx-auto mb-4" />
            <p className="text-slate-600 mb-4">No description available</p>
            <button
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
              onClick={handleRegenerate}
              disabled={loading}
            >
              {loading ? "Generating..." : "Generate Description"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DescriptionCard; 