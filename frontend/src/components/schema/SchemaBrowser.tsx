import React, { useState, useEffect } from 'react';
import { Database, Table, ChevronRight, ChevronDown, Search, RefreshCw, Eye } from 'lucide-react';
import { useDatabaseStore, useUIStore } from '../../stores';
import { schemaAPI } from '../../services/api';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { cn } from '../../utils/cn';
import toast from 'react-hot-toast';

interface SchemaNode {
  id: string;
  name: string;
  type: 'database' | 'table' | 'column';
  children?: SchemaNode[];
  expanded?: boolean;
  metadata?: any;
}

export const SchemaBrowser: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [schemaTree, setSchemaTree] = useState<SchemaNode[]>([]);
  const [selectedNode, setSelectedNode] = useState<SchemaNode | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const { databases, tables, loading, setTables, setLoading } = useDatabaseStore();
  const { activeDatabase } = useUIStore();

  useEffect(() => {
    if (activeDatabase) {
      loadTables(activeDatabase);
    }
  }, [activeDatabase]);

  const loadTables = async (databaseId: string) => {
    setLoading(true);
    try {
      const tablesData = await schemaAPI.getTables(databaseId);
      setTables(tablesData);
      buildSchemaTree(tablesData);
    } catch (error) {
      toast.error('Failed to load schema');
    } finally {
      setLoading(false);
    }
  };

  const buildSchemaTree = (tablesData: any[]) => {
    const tree: SchemaNode[] = tablesData.map(table => ({
      id: `table-${table.name}`,
      name: table.name,
      type: 'table',
      metadata: table,
      children: table.columns.map((column: any) => ({
        id: `column-${table.name}-${column.name}`,
        name: column.name,
        type: 'column',
        metadata: column,
      })),
      expanded: false,
    }));
    setSchemaTree(tree);
  };

  const handleNodeToggle = (nodeId: string) => {
    setSchemaTree(prev =>
      prev.map(node =>
        node.id === nodeId
          ? { ...node, expanded: !node.expanded }
          : node
      )
    );
  };

  const handleNodeSelect = (node: SchemaNode) => {
    setSelectedNode(node);
  };

  const handleRefresh = async () => {
    if (!activeDatabase) return;
    
    setRefreshing(true);
    try {
      await schemaAPI.refreshSchema(activeDatabase);
      await loadTables(activeDatabase);
      toast.success('Schema refreshed successfully');
    } catch (error) {
      toast.error('Failed to refresh schema');
    } finally {
      setRefreshing(false);
    }
  };

  const filteredTree = schemaTree.filter(node =>
    node.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    node.children?.some(child =>
      child.name.toLowerCase().includes(searchTerm.toLowerCase())
    )
  );

  if (!activeDatabase) {
    return (
      <div className="text-center py-8">
        <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-500">Select a database to explore its schema</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="grid grid-cols-12 gap-6">
        {/* Schema Tree */}
        <div className="col-span-8">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200">
            {/* Header */}
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Database className="w-5 h-5 text-gray-500" />
                  <h2 className="text-lg font-semibold text-gray-900">Schema Browser</h2>
                </div>
                <button
                  onClick={handleRefresh}
                  disabled={refreshing}
                  className="inline-flex items-center px-3 py-1 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors disabled:opacity-50"
                >
                  <RefreshCw className={cn('w-4 h-4 mr-1', refreshing && 'animate-spin')} />
                  Refresh
                </button>
              </div>
              
              {/* Search */}
              <div className="mt-4 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <input
                  type="text"
                  placeholder="Search tables and columns..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>

            {/* Tree */}
            <div className="max-h-96 overflow-y-auto">
              {loading ? (
                <div className="flex items-center justify-center p-8">
                  <LoadingSpinner />
                </div>
              ) : filteredTree.length === 0 ? (
                <div className="text-center py-8">
                  <Table className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">No tables found</p>
                </div>
              ) : (
                <div className="p-4">
                  {filteredTree.map((table) => (
                    <div key={table.id} className="mb-2">
                      <div
                        className={cn(
                          'flex items-center p-2 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors',
                          selectedNode?.id === table.id && 'bg-blue-50'
                        )}
                        onClick={() => handleNodeSelect(table)}
                      >
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleNodeToggle(table.id);
                          }}
                          className="mr-2 p-1 hover:bg-gray-200 rounded"
                        >
                          {table.expanded ? (
                            <ChevronDown className="w-4 h-4" />
                          ) : (
                            <ChevronRight className="w-4 h-4" />
                          )}
                        </button>
                        <Table className="w-4 h-4 text-blue-600 mr-2" />
                        <span className="font-medium text-gray-900">{table.name}</span>
                        <span className="ml-auto text-xs text-gray-500">
                          {table.metadata?.rowCount} rows
                        </span>
                      </div>
                      
                      {table.expanded && table.children && (
                        <div className="ml-6 mt-1 space-y-1">
                          {table.children.map((column) => (
                            <div
                              key={column.id}
                              className={cn(
                                'flex items-center p-2 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors',
                                selectedNode?.id === column.id && 'bg-blue-50'
                              )}
                              onClick={() => handleNodeSelect(column)}
                            >
                              <div className="w-4 h-4 mr-2" />
                              <div className="w-2 h-2 bg-gray-400 rounded-full mr-2" />
                              <span className="text-sm text-gray-700">{column.name}</span>
                              <span className="ml-auto text-xs text-gray-500 font-mono">
                                {column.metadata?.type}
                              </span>
                              {column.metadata?.primaryKey && (
                                <span className="ml-1 text-xs text-yellow-600">PK</span>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Details Panel */}
        <div className="col-span-4">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200">
            <div className="p-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Details</h3>
            </div>
            
            <div className="p-4">
              {selectedNode ? (
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-gray-900">{selectedNode.name}</h4>
                    <p className="text-sm text-gray-500 capitalize">{selectedNode.type}</p>
                  </div>
                  
                  {selectedNode.type === 'table' && selectedNode.metadata && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-500">Rows:</span>
                        <span className="font-medium">{selectedNode.metadata.rowCount}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-500">Size:</span>
                        <span className="font-medium">{selectedNode.metadata.size}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-500">Columns:</span>
                        <span className="font-medium">{selectedNode.metadata.columns.length}</span>
                      </div>
                      
                      <button className="w-full mt-4 inline-flex items-center justify-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 transition-colors">
                        <Eye className="w-4 h-4 mr-2" />
                        Preview Data
                      </button>
                    </div>
                  )}
                  
                  {selectedNode.type === 'column' && selectedNode.metadata && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-500">Type:</span>
                        <span className="font-medium font-mono">{selectedNode.metadata.type}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-500">Nullable:</span>
                        <span className="font-medium">{selectedNode.metadata.nullable ? 'Yes' : 'No'}</span>
                      </div>
                      {selectedNode.metadata.primaryKey && (
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-500">Primary Key:</span>
                          <span className="font-medium text-yellow-600">Yes</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Eye className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">Select a table or column to view details</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};