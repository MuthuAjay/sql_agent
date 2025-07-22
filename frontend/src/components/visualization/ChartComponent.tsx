import React from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  AreaChart,
  Area,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Download, Maximize2 } from 'lucide-react';
import { VisualizationResult } from '@/types/api';

interface ChartComponentProps {
  visualization: VisualizationResult;
}

const COLORS = [
  '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
  '#06B6D4', '#84CC16', '#F97316', '#EC4899', '#6B7280'
];

export function ChartComponent({ visualization }: ChartComponentProps) {
  const { chart_type, title, data, insights } = visualization;

  const downloadChart = () => {
    // This would need a more sophisticated implementation for actual chart export
    console.log('Downloading chart...', visualization);
  };

  const renderChart = () => {
    const chartData = data.labels.map((label, index) => ({
      name: label,
      ...data.datasets.reduce((acc, dataset, datasetIndex) => ({
        ...acc,
        [dataset.label || `Series ${datasetIndex + 1}`]: dataset.data[index] || 0,
      }), {}),
    }));

    switch (chart_type) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(156, 163, 175, 0.2)" />
              <XAxis dataKey="name" stroke="rgba(156, 163, 175, 0.8)" />
              <YAxis stroke="rgba(156, 163, 175, 0.8)" />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'rgba(17, 24, 39, 0.9)',
                  border: 'none',
                  borderRadius: '8px',
                  color: 'white',
                }}
              />
              <Legend />
              {data.datasets.map((dataset, index) => (
                <Bar
                  key={dataset.label || index}
                  dataKey={dataset.label || `Series ${index + 1}`}
                  fill={dataset.backgroundColor || COLORS[index % COLORS.length]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        );

      case 'line':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(156, 163, 175, 0.2)" />
              <XAxis dataKey="name" stroke="rgba(156, 163, 175, 0.8)" />
              <YAxis stroke="rgba(156, 163, 175, 0.8)" />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'rgba(17, 24, 39, 0.9)',
                  border: 'none',
                  borderRadius: '8px',
                  color: 'white',
                }}
              />
              <Legend />
              {data.datasets.map((dataset, index) => (
                <Line
                  key={dataset.label || index}
                  type="monotone"
                  dataKey={dataset.label || `Series ${index + 1}`}
                  stroke={dataset.borderColor || COLORS[index % COLORS.length]}
                  strokeWidth={2}
                  dot={{ fill: dataset.borderColor || COLORS[index % COLORS.length], strokeWidth: 2 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        );

      case 'pie':
        const pieData = chartData.map((item, index) => ({
          name: item.name,
          value: Object.values(item).find(val => typeof val === 'number') as number || 0,
        }));

        return (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        );

      case 'area':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={chartData}>
              <defs>
                {data.datasets.map((dataset, index) => (
                  <linearGradient key={index} id={`colorUv${index}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={COLORS[index % COLORS.length]} stopOpacity={0.8}/>
                    <stop offset="95%" stopColor={COLORS[index % COLORS.length]} stopOpacity={0.1}/>
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(156, 163, 175, 0.2)" />
              <XAxis dataKey="name" stroke="rgba(156, 163, 175, 0.8)" />
              <YAxis stroke="rgba(156, 163, 175, 0.8)" />
              <Tooltip />
              <Legend />
              {data.datasets.map((dataset, index) => (
                <Area
                  key={dataset.label || index}
                  type="monotone"
                  dataKey={dataset.label || `Series ${index + 1}`}
                  stroke={COLORS[index % COLORS.length]}
                  fillOpacity={1}
                  fill={`url(#colorUv${index})`}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        );

      case 'scatter':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(156, 163, 175, 0.2)" />
              <XAxis dataKey="name" stroke="rgba(156, 163, 175, 0.8)" />
              <YAxis stroke="rgba(156, 163, 175, 0.8)" />
              <Tooltip />
              <Legend />
              {data.datasets.map((dataset, index) => (
                <Scatter
                  key={dataset.label || index}
                  dataKey={dataset.label || `Series ${index + 1}`}
                  fill={COLORS[index % COLORS.length]}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        );

      default:
        return (
          <div className="flex items-center justify-center h-64 text-gray-500 dark:text-gray-400">
            Unsupported chart type: {chart_type}
          </div>
        );
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{title}</CardTitle>
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm" onClick={downloadChart}>
              <Download size={16} />
            </Button>
            <Button variant="outline" size="sm">
              <Maximize2 size={16} />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="mb-4">
          {renderChart()}
        </div>
        
        {insights && insights.length > 0 && (
          <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-md">
            <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">Chart Insights:</h4>
            <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
              {insights.map((insight, index) => (
                <li key={index}>• {insight}</li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}