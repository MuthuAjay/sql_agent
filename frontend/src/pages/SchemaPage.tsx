import React from "react";
import TablesListPanel from "../components/schema/TablesListPanel";
import TableDetailsPanel from "../components/schema/TableDetailsPanel";

const SchemaPage: React.FC = () => {
  return (
    <div className="flex h-full bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Left Sidebar - Tables List */}
      <div className="w-80 bg-white/80 backdrop-blur-sm border-r border-slate-200/60 shadow-lg">
        <TablesListPanel />
      </div>
      {/* Main Content - Table Details */}
      <div className="flex-1 flex flex-col">
        <TableDetailsPanel />
      </div>
    </div>
  );
};

export default SchemaPage;