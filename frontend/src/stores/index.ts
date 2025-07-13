import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Database, Table, QueryResult, QueryHistory, ChatMessage, UserPreferences } from '../types';
import { tableApi } from '../services/api';
import type { Table, TableSchema, SampleData } from '../types';

// UI Store
interface UIState {
  sidebarOpen: boolean;
  theme: 'light' | 'dark';
  activeDatabase: string | null;
  toggleSidebar: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
  setActiveDatabase: (databaseId: string | null) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      theme: 'light',
      activeDatabase: null,
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      setTheme: (theme) => set({ theme }),
      setActiveDatabase: (databaseId) => set({ activeDatabase: databaseId }),
    }),
    {
      name: 'ui-store',
    }
  )
);

// Database Store
interface DatabaseState {
  databases: Database[];
  tables: Table[];
  currentTable: Table | null;
  loading: boolean;
  error: string | null;
  setDatabases: (databases: Database[]) => void;
  setTables: (tables: Table[]) => void;
  setCurrentTable: (table: Table | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useDatabaseStore = create<DatabaseState>((set) => ({
  databases: [],
  tables: [],
  currentTable: null,
  loading: false,
  error: null,
  setDatabases: (databases) => set({ databases }),
  setTables: (tables) => set({ tables }),
  setCurrentTable: (table) => set({ currentTable: table }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
}));

// Query Store
interface QueryState {
  currentQuery: string;
  currentSql: string;
  queryResult: QueryResult | null;
  queryHistory: QueryHistory[];
  isExecuting: boolean;
  error: string | null;
  setCurrentQuery: (query: string) => void;
  setCurrentSql: (sql: string) => void;
  setQueryResult: (result: QueryResult | null) => void;
  setQueryHistory: (history: QueryHistory[]) => void;
  addToHistory: (query: QueryHistory) => void;
  setIsExecuting: (executing: boolean) => void;
  setError: (error: string | null) => void;
}

export const useQueryStore = create<QueryState>((set) => ({
  currentQuery: '',
  currentSql: '',
  queryResult: null,
  queryHistory: [],
  isExecuting: false,
  error: null,
  setCurrentQuery: (query) => set({ currentQuery: query }),
  setCurrentSql: (sql) => set({ currentSql: sql }),
  setQueryResult: (result) => set({ queryResult: result }),
  setQueryHistory: (history) => set({ queryHistory: history }),
  addToHistory: (query) => set((state) => ({ queryHistory: [query, ...state.queryHistory] })),
  setIsExecuting: (executing) => set({ isExecuting: executing }),
  setError: (error) => set({ error }),
}));

// Chat Store
interface ChatState {
  messages: ChatMessage[];
  isTyping: boolean;
  addMessage: (message: ChatMessage) => void;
  clearMessages: () => void;
  setIsTyping: (typing: boolean) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isTyping: false,
  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  clearMessages: () => set({ messages: [] }),
  setIsTyping: (typing) => set({ isTyping: typing }),
}));

// User Store
interface UserState {
  preferences: UserPreferences;
  favorites: any[];
  updatePreferences: (preferences: Partial<UserPreferences>) => void;
  setFavorites: (favorites: any[]) => void;
}

export const useUserStore = create<UserState>()(
  persist(
    (set) => ({
      preferences: {
        queryTimeout: 30,
        autoLimit: 1000,
        preferredViz: 'table',
        theme: 'light',
      },
      favorites: [],
      updatePreferences: (preferences) =>
        set((state) => ({
          preferences: { ...state.preferences, ...preferences },
        })),
      setFavorites: (favorites) => set({ favorites }),
    }),
    {
      name: 'user-store',
    }
  )
);

interface TableState {
  tables: Table[];
  selectedTable: string | null;
  tableSchema: TableSchema | null;
  sampleData: SampleData | null;
  loadingTables: boolean;
  loadingSchema: boolean;
  loadingDescription: boolean;
}

interface TableActions {
  fetchTables: (databaseId: string) => Promise<void>;
  fetchTableSchema: (databaseId: string, tableName: string) => Promise<void>;
  fetchSampleData: (databaseId: string, tableName: string) => Promise<void>;
  generateTableDescription: (databaseId: string, tableName: string, regenerate?: boolean) => Promise<void>;
  setSelectedTable: (tableName: string | null) => void;
  clearTableData: () => void;
}

export const useTableStore = create<TableState & TableActions>((set) => ({
  tables: [],
  selectedTable: null,
  tableSchema: null,
  sampleData: null,
  loadingTables: false,
  loadingSchema: false,
  loadingDescription: false,

  fetchTables: async (databaseId) => {
    set({ loadingTables: true });
    try {
      const { tables } = await tableApi.getTables(databaseId);
      set({ tables });
    } catch (e) {
      set({ tables: [] });
    } finally {
      set({ loadingTables: false });
    }
  },

  fetchTableSchema: async (databaseId, tableName) => {
    set({ loadingSchema: true });
    try {
      const schema = await tableApi.getTableSchema(databaseId, tableName);
      set({ tableSchema: schema });
    } catch (e) {
      set({ tableSchema: null });
    } finally {
      set({ loadingSchema: false });
    }
  },

  fetchSampleData: async (databaseId, tableName) => {
    try {
      const sample = await tableApi.getSampleData(databaseId, tableName);
      set({ sampleData: sample });
    } catch (e) {
      set({ sampleData: null });
    }
  },

  generateTableDescription: async (databaseId, tableName, regenerate = false) => {
    set({ loadingDescription: true });
    try {
      const { description } = await tableApi.generateDescription(databaseId, tableName, regenerate);
      set({ tableSchema: { ...useTableStore.getState().tableSchema, description } });
    } catch (e) {
      // Optionally handle error
    } finally {
      set({ loadingDescription: false });
    }
  },

  setSelectedTable: (tableName) => set({ selectedTable: tableName }),
  clearTableData: () => set({ tableSchema: null, sampleData: null }),
}));