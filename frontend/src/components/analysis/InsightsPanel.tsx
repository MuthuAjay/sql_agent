import React from 'react';
import { TrendingUp, TrendingDown, AlertTriangle, Lightbulb, Target, Award } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { AnalysisResult } from '@/types/api';
import { cn } from '@/lib/utils';

interface InsightsPanelProps {
  analysis: AnalysisResult;
}

export function InsightsPanel({ analysis }: InsightsPanelProps) {
  const {
    summary,
    insights,
    anomalies,
    trends,
    recommendations,
    data_quality_score,
    confidence_score,
  } = analysis;

  const getImpactColor = (impact: 'high' | 'medium' | 'low') => {
    switch (impact) {
      case 'high': return 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
      case 'medium': return 'text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
      case 'low': return 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
    }
  };

  const getTrendIcon = (trend_type: string) => {
    switch (trend_type) {
      case 'increasing': return <TrendingUp size={16} className="text-green-500" />;
      case 'decreasing': return <TrendingDown size={16} className="text-red-500" />;
      default: return <TrendingUp size={16} className="text-gray-500" />;
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Summary Stats */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Award size={20} className="text-blue-600" />
            <span>Data Summary</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600 dark:text-gray-400">Total Rows</span>
              <p className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                {summary.total_rows.toLocaleString()}
              </p>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">Data Quality</span>
              <p className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                {(data_quality_score * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">Confidence</span>
              <p className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                {(confidence_score * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">Missing Values</span>
              <p className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                {Object.values(summary.missing_values || {}).reduce((sum, count) => sum + count, 0)}
              </p>
            </div>
          </div>
          
          {summary.data_quality_issues && summary.data_quality_issues.length > 0 && (
            <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded-md">
              <h4 className="font-medium text-orange-900 dark:text-orange-100 mb-2">Data Quality Issues:</h4>
              <ul className="text-sm text-orange-800 dark:text-orange-200 space-y-1">
                {summary.data_quality_issues.map((issue, index) => (
                  <li key={index}>• {issue}</li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Key Insights */}
      {insights && insights.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Lightbulb size={20} className="text-yellow-600" />
              <span>Key Insights</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {insights.slice(0, 3).map((insight, index) => (
              <div
                key={index}
                className={cn('p-3 rounded-md border', getImpactColor(insight.impact))}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium mb-1">{insight.title}</h4>
                    <p className="text-sm opacity-90">{insight.description}</p>
                  </div>
                  <span className="text-xs px-2 py-1 rounded-full bg-white dark:bg-gray-800 opacity-75">
                    {(insight.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Anomalies */}
      {anomalies && anomalies.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle size={20} className="text-red-600" />
              <span>Anomalies Detected</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {anomalies.slice(0, 3).map((anomaly, index) => (
              <div
                key={index}
                className={cn('p-3 rounded-md border', getImpactColor(anomaly.severity as any))}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium mb-1">{anomaly.column} - {anomaly.type}</h4>
                    <p className="text-sm opacity-90">{anomaly.description}</p>
                    <p className="text-xs mt-1 opacity-75">
                      Affected rows: {anomaly.affected_rows}
                    </p>
                  </div>
                  <span className="text-xs px-2 py-1 rounded-full bg-white dark:bg-gray-800 opacity-75">
                    {anomaly.severity}
                  </span>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Trends */}
      {trends && trends.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp size={20} className="text-green-600" />
              <span>Trends Analysis</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {trends.slice(0, 3).map((trend, index) => (
              <div
                key={index}
                className="p-3 bg-gray-50 dark:bg-gray-800/50 rounded-md border border-gray-200 dark:border-gray-700"
              >
                <div className="flex items-start space-x-2">
                  {getTrendIcon(trend.trend_type)}
                  <div className="flex-1">
                    <h4 className="font-medium mb-1">{trend.column}</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{trend.description}</p>
                    <span className="text-xs text-gray-500 dark:text-gray-500">
                      Confidence: {(trend.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target size={20} className="text-blue-600" />
              <span>Recommendations</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {recommendations.map((recommendation, index) => (
                <div
                  key={index}
                  className={cn('p-4 rounded-md border', getImpactColor(recommendation.priority as any))}
                >
                  <h4 className="font-medium mb-2">{recommendation.title}</h4>
                  <p className="text-sm opacity-90 mb-2">{recommendation.description}</p>
                  <div className="text-xs space-y-1">
                    <p><strong>Action:</strong> {recommendation.action}</p>
                    <p><strong>Expected Impact:</strong> {recommendation.expected_impact}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}