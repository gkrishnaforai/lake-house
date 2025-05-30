// Types
export interface TableInfo {
  name: string;
  description?: string;
  columns: ColumnInfo[];
  row_count?: number;
  last_updated?: string;
  s3_location?: string;
}

export interface ColumnInfo {
  name: string;
  type: string;
  description?: string;
}

export interface DatabaseCatalog {
  database_name: string;
  tables: TableInfo[];
  description?: string;
  metadata?: Record<string, any>;
}

export interface FileMetadata {
  file_name: string;
  file_type: string;
  s3_path: string;
  schema_path: string;
  created_at: string;
  size: number;
}

export interface UploadResponse {
  status: string;
  message: string;
  file_name: string;
  s3_path: string;
  table_name?: string;
  error?: string;
  details?: Record<string, any>;
}

export interface DescriptiveQueryResponse {
  status: string;
  query?: string;
  results?: any[];
  message?: string;
}

export interface ChatRequest {
  message: string;
  schema?: any;
  sample_data?: any;
}

export interface ChatResponse {
  status: string;
  response: string;
  message: string;
}

// API Configuration
const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

// Helper function for API calls
async function apiCall<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const defaultHeaders = {
    'X-User-Id': 'test_user',
    ...(options.headers || {}),
  };

  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: defaultHeaders,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || 'API call failed');
  }

  return res.json();
}

// Catalog APIs
export async function getTables(): Promise<TableInfo[]> {
  return apiCall<TableInfo[]>('/api/catalog/tables');
}

export async function getDatabaseCatalog(): Promise<DatabaseCatalog> {
  return apiCall<DatabaseCatalog>('/api/catalog');
}

export async function getCatalogFiles(): Promise<FileMetadata[]> {
  return apiCall<FileMetadata[]>('/api/catalog/files');
}

// File Operations
export async function uploadFile(tableName: string, file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  return apiCall<UploadResponse>(`/api/catalog/tables/${tableName}/files`, {
    method: 'POST',
    body: formData,
  });
}

export async function getTableFiles(tableName: string): Promise<FileMetadata[]> {
  return apiCall<FileMetadata[]>(`/api/catalog/tables/${tableName}/files`);
}

export async function getFileDetails(s3Path: string): Promise<FileMetadata> {
  return apiCall<FileMetadata>(`/api/files/${s3Path}/details`);
}

export async function getFilePreview(s3Path: string, limit: number = 10): Promise<any[]> {
  return apiCall<any[]>(`/api/files/${s3Path}/preview?limit=${limit}`);
}

export async function deleteFile(s3Path: string): Promise<{ message: string }> {
  return apiCall<{ message: string }>(`/api/files/${s3Path}`, {
    method: 'DELETE',
  });
}

// Query Operations
export async function runDescriptiveQuery(
  query: string,
  tableName: string
): Promise<DescriptiveQueryResponse> {
  return apiCall<DescriptiveQueryResponse>('/api/catalog/descriptive_query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query, table_name: tableName }),
  });
}

// Table Operations
export async function createUserTable(
  userId: string,
  tableName: string,
  schema: Record<string, any>,
  originalDataLocation: string,
  parquetDataLocation: string,
  options: {
    fileFormat?: string;
    partitionKeys?: Array<{ name: string; type: string }>;
    metadata?: Record<string, any>;
    compression?: string;
    encryption?: string;
  } = {}
): Promise<Record<string, any>> {
  return apiCall<Record<string, any>>(`/api/v1/users/${userId}/tables`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      table_name: tableName,
      schema,
      original_data_location: originalDataLocation,
      parquet_data_location: parquetDataLocation,
      ...options,
    }),
  });
}

export async function processTable(tableName: string): Promise<any> {
  return apiCall<any>(`/api/catalog/tables/${tableName}/process`);
}

// Health Check
export async function checkHealth(): Promise<{ status: string; timestamp: string }> {
  return apiCall<{ status: string; timestamp: string }>('/health');
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  return apiCall<ChatResponse>('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });
} 