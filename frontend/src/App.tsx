import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { AppLayout } from '@/components/layout/AppLayout';
import { QueryPage } from '@/pages/QueryPage';
import { SQLEditorPage } from '@/pages/SQLEditorPage';
import { SchemaPage } from '@/pages/SchemaPage';
import { AnalyticsPage } from '@/pages/AnalyticsPage';
import { HistoryPage } from '@/pages/HistoryPage';
import { PerformancePage } from '@/pages/PerformancePage';
import { HealthPage } from '@/pages/HealthPage';
import { SettingsPage } from '@/pages/SettingsPage';
import { ErrorBoundary } from '@/components/ErrorBoundary';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <QueryClientProvider client={queryClient}>
          <Router>
            <div className="h-screen overflow-hidden">
              <Routes>
                <Route path="/" element={<AppLayout />}>
                  <Route index element={<QueryPage />} />
                  <Route path="sql" element={<SQLEditorPage />} />
                  <Route path="schema" element={<SchemaPage />} />
                  <Route path="analytics" element={<AnalyticsPage />} />
                  <Route path="history" element={<HistoryPage />} />
                  <Route path="performance" element={<PerformancePage />} />
                  <Route path="health" element={<HealthPage />} />
                  <Route path="settings" element={<SettingsPage />} />
                </Route>
              </Routes>
            </div>
            <Toaster 
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: 'var(--toast-bg, #363636)',
                  color: 'var(--toast-color, #fff)',
                },
                success: {
                  iconTheme: {
                    primary: '#10B981',
                    secondary: '#fff',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#EF4444',
                    secondary: '#fff',
                  },
                },
              }}
            />
          </Router>
        </QueryClientProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;