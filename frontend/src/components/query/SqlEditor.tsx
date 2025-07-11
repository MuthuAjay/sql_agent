import React, { useRef, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { Play, Code, Wand2 } from 'lucide-react';
import { sqlAPI } from '../../services/api';
import { useUIStore } from '../../stores';
import toast from 'react-hot-toast';

interface SqlEditorProps {
  value: string;
  onChange: (value: string) => void;
  onExecute: () => void;
  isExecuting: boolean;
}

export const SqlEditor: React.FC<SqlEditorProps> = ({
  value,
  onChange,
  onExecute,
  isExecuting,
}) => {
  const editorRef = useRef<any>(null);
  const { theme, activeDatabase } = useUIStore();

  const handleEditorDidMount = (editor: any) => {
    editorRef.current = editor;
    
    // Add keyboard shortcuts
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
      onExecute();
    });
  };

  const handleFormat = async () => {
    if (!value.trim()) return;

    try {
      const result = await sqlAPI.format(value);
      onChange(result.formatted);
      toast.success('SQL formatted successfully');
    } catch (error) {
      toast.error('Failed to format SQL');
    }
  };

  const handleOptimize = async () => {
    if (!value.trim() || !activeDatabase) return;

    try {
      const result = await sqlAPI.optimize(value, activeDatabase);
      if (result.suggestions.length > 0) {
        toast.success(`Found ${result.suggestions.length} optimization suggestions`);
        // You could show these suggestions in a modal or panel
      } else {
        toast.success('No optimization suggestions found');
      }
    } catch (error) {
      toast.error('Failed to analyze SQL');
    }
  };

  return (
    <div className="border-t border-gray-200">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-4 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center space-x-2">
          <Code className="w-4 h-4 text-gray-500" />
          <span className="text-sm font-medium text-gray-700">SQL Editor</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={handleFormat}
            className="inline-flex items-center px-3 py-1 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors"
          >
            <Code className="w-3 h-3 mr-1" />
            Format
          </button>
          
          <button
            onClick={handleOptimize}
            disabled={!activeDatabase}
            className="inline-flex items-center px-3 py-1 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors disabled:opacity-50"
          >
            <Wand2 className="w-3 h-3 mr-1" />
            Optimize
          </button>
        </div>
      </div>

      {/* Editor */}
      <div className="h-64">
        <Editor
          height="100%"
          defaultLanguage="sql"
          value={value}
          onChange={(value) => onChange(value || '')}
          onMount={handleEditorDidMount}
          theme={theme === 'dark' ? 'vs-dark' : 'vs-light'}
          options={{
            minimap: { enabled: false },
            fontSize: 14,
            lineNumbers: 'on',
            wordWrap: 'on',
            automaticLayout: true,
            scrollBeyondLastLine: false,
            formatOnPaste: true,
            formatOnType: true,
            suggest: {
              showKeywords: true,
              showSnippets: true,
            },
          }}
        />
      </div>

      <div className="p-4 bg-gray-50 border-t border-gray-200 text-xs text-gray-500">
        Press <kbd className="px-2 py-1 bg-gray-100 rounded">⌘</kbd> + <kbd className="px-2 py-1 bg-gray-100 rounded">Enter</kbd> to execute
      </div>
    </div>
  );
};