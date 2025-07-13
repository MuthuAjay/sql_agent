import React from "react";
import TableHeaderCard from "./TableHeaderCard";
import ColumnsCard from "./ColumnsCard";
import SampleDataCard from "./SampleDataCard";
import DescriptionCard from "./DescriptionCard";
import IndexesCard from "./IndexesCard";
import RelationshipsCard from "./RelationshipsCard";
import EmptyState from "./EmptyState";
import { useTableStore, useUIStore } from "../../stores";
import LoadingSpinner from "../common/LoadingSpinner";

const TableDetailsPanel: React.FC = () => {
  const selectedTable = useTableStore((s) => s.selectedTable);
  const tableSchema = useTableStore((s) => s.tableSchema);
  const sampleData = useTableStore((s) => s.sampleData);
  const loadingSchema = useTableStore((s) => s.loadingSchema);
  const loadingDescription = useTableStore((s) => s.loadingDescription);
  const activeDatabase = useUIStore((s) => s.activeDatabase);
  const fetchTableSchema = useTableStore((s) => s.fetchTableSchema);
  const fetchSampleData = useTableStore((s) => s.fetchSampleData);
  const generateTableDescription = useTableStore((s) => s.generateTableDescription);

  React.useEffect(() => {
    if (activeDatabase && selectedTable) {
      fetchTableSchema(activeDatabase, selectedTable);
      fetchSampleData(activeDatabase, selectedTable);
      generateTableDescription(activeDatabase, selectedTable, false);
    }
  }, [activeDatabase, selectedTable, fetchTableSchema, fetchSampleData, generateTableDescription]);

  if (!selectedTable) {
    return <EmptyState />;
  }

  if (loadingSchema) {
    return <LoadingSpinner />;
  }

  if (!tableSchema) {
    return <div className="text-center text-slate-500 py-8">No schema data available.</div>;
  }

  return (
    <div className="flex-1 p-6 overflow-auto">
      <div className="max-w-6xl mx-auto space-y-6">
        <TableHeaderCard
          tableName={tableSchema.name}
          columnsCount={tableSchema.columns?.length || 0}
          rowsCount={tableSchema.rowCount || 0}
        />
        {/* Schema Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ColumnsCard columns={tableSchema.columns || []} />
          <SampleDataCard data={sampleData} />
        </div>
        {/* Description Card */}
        <DescriptionCard tableName={tableSchema.name} />
        {/* Indexes and Relationships Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <IndexesCard indexes={tableSchema.indexes || []} />
          <RelationshipsCard relationships={tableSchema.relationships || []} />
        </div>
      </div>
    </div>
  );
};

export default TableDetailsPanel; 