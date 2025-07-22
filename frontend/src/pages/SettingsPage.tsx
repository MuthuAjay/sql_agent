import React, { useState } from 'react';
import { Settings, Database, Bell, Shield, Palette, Download, Upload } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { useTheme } from '@/contexts/ThemeContext';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import toast from 'react-hot-toast';

interface SettingsData {
  api: {
    baseUrl: string;
    timeout: number;
    maxRetries: number;
  };
  query: {
    defaultDatabase: string;
    maxResults: number;
    includeAnalysis: boolean;
    includeVisualization: boolean;
  };
  ui: {
    itemsPerPage: number;
    autoSave: boolean;
    showQueryTime: boolean;
  };
  notifications: {
    queryComplete: boolean;
    queryError: boolean;
    systemAlerts: boolean;
  };
}

const defaultSettings: SettingsData = {
  api: {
    baseUrl: 'http://localhost:8000',
    timeout: 30000,
    maxRetries: 2,
  },
  query: {
    defaultDatabase: 'default',
    maxResults: 100,
    includeAnalysis: true,
    includeVisualization: true,
  },
  ui: {
    itemsPerPage: 50,
    autoSave: true,
    showQueryTime: true,
  },
  notifications: {
    queryComplete: true,
    queryError: true,
    systemAlerts: true,
  },
};

export function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const [settings, setSettings] = useLocalStorage<SettingsData>('app-settings', defaultSettings);
  const [tempSettings, setTempSettings] = useState<SettingsData>(settings);
  const [hasChanges, setHasChanges] = useState(false);

  const handleSettingChange = (section: keyof SettingsData, key: string, value: any) => {
    const newSettings = {
      ...tempSettings,
      [section]: {
        ...tempSettings[section],
        [key]: value,
      },
    };
    setTempSettings(newSettings);
    setHasChanges(JSON.stringify(newSettings) !== JSON.stringify(settings));
  };

  const saveSettings = () => {
    setSettings(tempSettings);
    setHasChanges(false);
    toast.success('Settings saved successfully');
  };

  const resetSettings = () => {
    if (confirm('Are you sure you want to reset all settings to defaults?')) {
      setTempSettings(defaultSettings);
      setSettings(defaultSettings);
      setHasChanges(false);
      toast.success('Settings reset to defaults');
    }
  };

  const exportSettings = () => {
    const dataStr = JSON.stringify(settings, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = 'sql-agent-settings.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    toast.success('Settings exported successfully');
  };

  const importSettings = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importedSettings = JSON.parse(e.target?.result as string);
        setTempSettings(importedSettings);
        setHasChanges(true);
        toast.success('Settings imported successfully');
      } catch (error) {
        toast.error('Invalid settings file');
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="h-full bg-gray-50 dark:bg-gray-900 p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Settings size={24} className="text-blue-600" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Settings</h1>
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="file"
            accept=".json"
            onChange={importSettings}
            className="hidden"
            id="import-settings"
          />
          <label htmlFor="import-settings">
            <Button variant="outline" size="sm" as="span">
              <Upload size={16} />
              Import
            </Button>
          </label>
          
          <Button variant="outline" size="sm" onClick={exportSettings}>
            <Download size={16} />
            Export
          </Button>
          
          {hasChanges && (
            <>
              <Button variant="outline" size="sm" onClick={() => setTempSettings(settings)}>
                Cancel
              </Button>
              <Button onClick={saveSettings}>
                Save Changes
              </Button>
            </>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* API Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Database size={20} />
              <span>API Configuration</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Input
              label="API Base URL"
              value={tempSettings.api.baseUrl}
              onChange={(e) => handleSettingChange('api', 'baseUrl', e.target.value)}
              placeholder="http://localhost:8000"
            />
            
            <Input
              label="Request Timeout (ms)"
              type="number"
              value={tempSettings.api.timeout}
              onChange={(e) => handleSettingChange('api', 'timeout', parseInt(e.target.value))}
              helpText="Maximum time to wait for API responses"
            />
            
            <Input
              label="Max Retries"
              type="number"
              value={tempSettings.api.maxRetries}
              onChange={(e) => handleSettingChange('api', 'maxRetries', parseInt(e.target.value))}
              helpText="Number of retry attempts for failed requests"
            />
          </CardContent>
        </Card>

        {/* Query Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Query Preferences</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Input
              label="Default Database"
              value={tempSettings.query.defaultDatabase}
              onChange={(e) => handleSettingChange('query', 'defaultDatabase', e.target.value)}
              placeholder="default"
            />
            
            <Input
              label="Max Results per Query"
              type="number"
              value={tempSettings.query.maxResults}
              onChange={(e) => handleSettingChange('query', 'maxResults', parseInt(e.target.value))}
              helpText="Maximum number of rows to return"
            />
            
            <div className="space-y-3">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={tempSettings.query.includeAnalysis}
                  onChange={(e) => handleSettingChange('query', 'includeAnalysis', e.target.checked)}
                  className="rounded border-gray-300"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Include analysis by default
                </span>
              </label>
              
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={tempSettings.query.includeVisualization}
                  onChange={(e) => handleSettingChange('query', 'includeVisualization', e.target.checked)}
                  className="rounded border-gray-300"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Include visualizations by default
                </span>
              </label>
            </div>
          </CardContent>
        </Card>

        {/* UI Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Palette size={20} />
              <span>User Interface</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Theme
              </label>
              <select
                value={theme}
                onChange={(e) => setTheme(e.target.value as any)}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md px-3 py-2 text-sm"
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="system">System</option>
              </select>
            </div>
            
            <Input
              label="Items per Page"
              type="number"
              value={tempSettings.ui.itemsPerPage}
              onChange={(e) => handleSettingChange('ui', 'itemsPerPage', parseInt(e.target.value))}
              helpText="Number of items to show in paginated lists"
            />
            
            <div className="space-y-3">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={tempSettings.ui.autoSave}
                  onChange={(e) => handleSettingChange('ui', 'autoSave', e.target.checked)}
                  className="rounded border-gray-300"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Auto-save queries
                </span>
              </label>
              
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={tempSettings.ui.showQueryTime}
                  onChange={(e) => handleSettingChange('ui', 'showQueryTime', e.target.checked)}
                  className="rounded border-gray-300"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Show query execution time
                </span>
              </label>
            </div>
          </CardContent>
        </Card>

        {/* Notification Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Bell size={20} />
              <span>Notifications</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={tempSettings.notifications.queryComplete}
                  onChange={(e) => handleSettingChange('notifications', 'queryComplete', e.target.checked)}
                  className="rounded border-gray-300"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Query completion notifications
                </span>
              </label>
              
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={tempSettings.notifications.queryError}
                  onChange={(e) => handleSettingChange('notifications', 'queryError', e.target.checked)}
                  className="rounded border-gray-300"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Query error notifications
                </span>
              </label>
              
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={tempSettings.notifications.systemAlerts}
                  onChange={(e) => handleSettingChange('notifications', 'systemAlerts', e.target.checked)}
                  className="rounded border-gray-300"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  System alerts
                </span>
              </label>
            </div>
          </CardContent>
        </Card>

        {/* Advanced Settings */}
        <Card className="xl:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Shield size={20} />
              <span>Advanced Settings</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white mb-3">Security</h4>
                <div className="space-y-3">
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      defaultChecked={true}
                      className="rounded border-gray-300"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      Enable request logging
                    </span>
                  </label>
                  
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      defaultChecked={false}
                      className="rounded border-gray-300"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      Allow dangerous queries
                    </span>
                  </label>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white mb-3">Performance</h4>
                <div className="space-y-3">
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      defaultChecked={true}
                      className="rounded border-gray-300"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      Enable query caching
                    </span>
                  </label>
                  
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      defaultChecked={true}
                      className="rounded border-gray-300"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      Optimize for mobile
                    </span>
                  </label>
                </div>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
              <Button
                variant="destructive"
                onClick={resetSettings}
                className="mr-2"
              >
                Reset to Defaults
              </Button>
              
              <Button
                variant="outline"
                onClick={() => {
                  localStorage.clear();
                  toast.success('Local storage cleared');
                }}
              >
                Clear All Data
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}