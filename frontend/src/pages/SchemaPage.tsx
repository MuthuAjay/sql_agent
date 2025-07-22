import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Database, Table, Search, ChevronRight, ChevronDown, Eye } from 'lucide-react';
import { api } from '@/lib/api';
import { Input } from '@/components/ui/Input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { TableInfo } from '@/types/api';
import { cn } from '@/lib/utils';

export function SchemaPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedTables, setExpandedTables] = useState<Set<string>>(new Set());
  const [selectedTable, setSelectedTable] = useState<string | null>(null);

  const { data: schema, isLoading, error } = useQuery({
    queryKey: ['schema'],
    queryFn: () => api.schema.get(),
  });

  const { data: selectedTableData } = useQuery({
    queryKey: ['table', selectedTable],
    queryFn: () => api.schema.getTable(selectedTable!),
    enabled: !!selectedTable,
  });

  const filteredTables = schema?.tables?.filter((table) =>
    table.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    table.columns.some((col) => col.name.toLowerCase().includes(searchQuery.toLowerCase()))
  ) || [];

  const toggleTable = (tableName: string) => {
    const newExpanded = new Set(expandedTables);
    if (newExpanded.has(tableName)) {
      newExpanded.delete(tableName);
    } else {
      newExpanded.add(tableName);
    }
    setExpandedTables(newExpanded);
  };

  const getColumnTypeColor = (type: string) => {
    if (type.includes('int') || type.includes('number') || type.includes('decimal')) {
      return 'text-blue-600 dark:text-blue-400';
    }
    if (type.includes('varchar') || type.includes('text') || type.includes('string')) {
      return 'text-green-600 dark:text-green-400';
    }
    if (type.includes('date') || type.includes('time')) {
      return 'text-purple-600 dark:text-purple-400';
    }
    if (type.includes('bool')) {
      return 'text-orange-600 dark:text-orange-400';
    }
    return 'text-gray-600 dark:text-gray-400';
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <p className="text-red-600 dark:text-red-400 mb-2">Failed to load schema</p>
          <p className="text-gray-500 dark:text-gray-400">Please check your database connection</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex">
      {/* Schema Explorer */}
      <div className="w-1/2 border-r border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-2">
              <Database size={24} className="text-blue-600" />
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                {schema?.database_name || 'Database Schema'}
              </h1>
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              {schema?.total_tables} tables, {schema?.total_columns} columns
            </div>
          </div>

          <div className="mb-4">
            <div className="relative">
              <Search size={18} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <Input
                type="text"
                placeholder="Search tables and columns..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>

          <div className="space-y-2 max-h-[calc(100vh-240px)] overflow-auto">
            {filteredTables.map((table) => (
              <div key={table.name} className="border border-gray-200 dark:border-gray-700 rounded-md">
                <div
                  className="flex items-center justify-between p-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
                  onClick={() => toggleTable(table.name)}
                >
                  <div className="flex items-center space-x-2">
                    {expandedTables.has(table.name) ? (
                      <ChevronDown size={16} className="text-gray-500" />
                    ) : (
                      <ChevronRight size={16} className="text-gray-500" />
                    )}
                    <Table size={16} className="text-gray-600 dark:text-gray-400" />
                    <span className="font-medium text-gray-900 dark:text-white">{table.name}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {table.columns.length} columns
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedTable(table.name);
                      }}
                    >
                      <Eye size={14} />
                    </Button>
                  </div>
                </div>

                {expandedTables.has(table.name) && (
                  <div className="border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/30">
                    <div className="p-3 space-y-2">
                      {table.columns.map((column) => (
                        <div key={column.name} className="flex items-center justify-between text-sm">
                          <div className="flex items-center space-x-2">
                            <span className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500" />
                            <span className="font-medium text-gray-700 dark:text-gray-300">
                              {column.name}
                            </span>
                            {column.primary_key && (
                              <span className="px-1.5 py-0.5 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300 text-xs rounded">
                                PK
                              </span>
                            )}
                            {column.foreign_key && (
                              <span className="px-1.5 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 text-xs rounded">
                                FK
                              </span>
                            )}
                          </div>
                          <div className="flex items-center space-x-2">
                            <span className={cn('text-xs font-medium', getColumnTypeColor(column.type))}>
                              {column.type}
                            </span>
                            {!column.nullable && (
                              <span className="text-xs text-red-500 dark:text-red-400">NOT NULL</span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Table Details */}
      <div className="w-1/2 bg-gray-50 dark:bg-gray-900 p-6">
        {selectedTableData ? (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Table size={20} />
                  <span>{selectedTableData.name}</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Columns</span>
                    <p className="font-semibold text-gray-900 dark:text-gray-100">
                      {selectedTableData.columns.length}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Rows</span>
                    <p className="font-semibold text-gray-900 dark:text-gray-100">
                      {selectedTableData.row_count?.toLocaleString() || 'N/A'}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Size</span>
                    <p className="font-semibold text-gray-900 dark:text-gray-100">
                      {selectedTableData.size || 'N/A'}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Relationships</span>
                    <p className="font-semibold text-gray-900 dark:text-gray-100">
                      {selectedTableData.relationships.length}
                    </p>
                  </div>
                </div>

                {selectedTableData.description && (
                  <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-md">
                    <p className="text-sm text-blue-800 dark:text-blue-200">
                      {selectedTableData.description}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Column Details */}
            <Card>
              <CardHeader>
                <CardTitle>Column Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {selectedTableData.columns.map((column) => (
                    <div
                      key={column.name}
                      className="p-3 bg-white dark:bg-gray-800 rounded-md border border-gray-200 dark:border-gray-700"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-gray-900 dark:text-white">
                          {column.name}
                        </span>
                        <div className="flex items-center space-x-2">
                          <span className={cn('text-sm font-medium', getColumnTypeColor(column.type))}>
                            {column.type}
                          </span>
                          {column.primary_key && (
                            <span className="px-2 py-1 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300 text-xs rounded">
                              Primary Key
                            </span>
                          )}
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-400">
                        <span>Nullable: {column.nullable ? 'Yes' : 'No'}</span>
                        {column.foreign_key && (
                          <span>
                            FK → {column.foreign_key.referenced_table}.{column.foreign_key.referenced_column}
                          </span>
                        )}
                      </div>

                      {column.description && (
                        <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                          {column.description}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Relationships */}
            {selectedTableData.relationships.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Relationships</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {selectedTableData.relationships.map((rel, index) => (
                      <div
                        key={index}
                        className="p-3 bg-gray-100 dark:bg-gray-800 rounded-md text-sm"
                      >
                        <div className="flex items-center space-x-2">
                          <span className="font-medium">{rel.from_table}.{rel.from_column}</span>
                          <span className="text-gray-500">→</span>
                          <span className="font-medium">{rel.to_table}.{rel.to_column}</span>
                          <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 text-xs rounded">
                            {rel.relationship_type.replace('_', ' ')}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
            <div className="text-center">
              <Table size={48} className="mx-auto mb-4 opacity-50" />
              <p>Select a table to view its details</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}