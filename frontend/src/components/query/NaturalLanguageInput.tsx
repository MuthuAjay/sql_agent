import React, { useState } from 'react';
import { Send, Lightbulb } from 'lucide-react';
import { cn } from '../../utils/cn';

interface NaturalLanguageInputProps {
  value: string;
  onChange: (value: string) => void;
  onExecute: () => void;
  isExecuting: boolean;
}

const suggestions = [
  "Show me total sales by month for the last year",
  "What are the top 10 customers by revenue?",
  "Find all products with low inventory",
  "Show me average order value by region",
  "Which employees have the highest sales?",
];

export const NaturalLanguageInput: React.FC<NaturalLanguageInputProps> = ({
  value,
  onChange,
  onExecute,
  isExecuting,
}) => {
  const [showSuggestions, setShowSuggestions] = useState(false);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      onExecute();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    onChange(suggestion);
    setShowSuggestions(false);
  };

  return (
    <div className="p-6">
      <div className="relative">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => setShowSuggestions(true)}
          placeholder="Ask me anything about your data... 
          
Examples:
• Show me sales trends over the last 6 months
• Which products are performing best?
• What's the customer distribution by region?"
          className="w-full min-h-[120px] p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
          disabled={isExecuting}
        />
        
        <div className="absolute bottom-3 right-3 flex items-center space-x-2">
          <button
            onClick={() => setShowSuggestions(!showSuggestions)}
            className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
            title="Show suggestions"
          >
            <Lightbulb className="w-4 h-4" />
          </button>
          
          <button
            onClick={onExecute}
            disabled={isExecuting || !value.trim()}
            className={cn(
              'p-2 rounded-lg transition-colors',
              isExecuting || !value.trim()
                ? 'text-gray-400 cursor-not-allowed'
                : 'text-blue-600 hover:bg-blue-50'
            )}
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Suggestions */}
      {showSuggestions && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
            <Lightbulb className="w-4 h-4 mr-1" />
            Suggestions
          </h4>
          <div className="space-y-2">
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(suggestion)}
                className="block w-full text-left text-sm text-gray-600 hover:text-blue-600 hover:bg-white p-2 rounded transition-colors"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="mt-4 text-xs text-gray-500">
        Press <kbd className="px-2 py-1 bg-gray-100 rounded">⌘</kbd> + <kbd className="px-2 py-1 bg-gray-100 rounded">Enter</kbd> to execute
      </div>
    </div>
  );
};