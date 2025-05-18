// API Response Types
export interface ApiResponse<T> {
  data: T;
  status: 'success' | 'error';
  message?: string;
}

// Catalog Types
export interface CatalogResponse {
  database_name: string;
  description?: string;
  tables: TableInfo[];
}

export interface TableInfo {
  name: string;
  description?: string;
  location?: string;
  last_updated?: string;
  columns?: ColumnInfo[];
}

export interface ColumnInfo {
  name: string;
  type: string;
  description?: string;
}

// File Management Types
export interface FileInfo {
  file_name: string;
  s3_path: string;
  status: string;
  table_name?: string;
  error?: string;
  details?: any;
  size?: number;
  last_modified?: string;
}

export interface FileDetails {
  file_name: string;
  file_type: string;
  size: number;
  last_modified: string;
}

export interface FileUploadResponse {
  status: string;
  message: string;
  file_name: string;
  s3_path: string;
  table_name?: string;
  error?: string;
  details?: any;
  schema?: Record<string, string>;
  sample_data?: any[];
}

// Data Quality Types
export interface DataQualityResponse {
  table_name: string;
  overall_score: number;
  columns: Array<{
    name: string;
    type: string;
    quality_metrics: {
      completeness: number;
      uniqueness: number;
      accuracy: number;
    };
  }>;
}

// Audit Trail Types
export interface AuditTrailResponse {
  audit_metadata: {
    created_at: string;
    created_by: string;
    updated_at: string;
    updated_by: string;
  };
  conversion_history: Array<{
    from_format: string;
    to_format: string;
    converted_by: string;
    converted_at: string;
  }>;
}

// Chat Types
export interface ChatMessage {
  type: 'user' | 'assistant' | 'error';
  content: string;
  timestamp: string;
  sql?: string;
  data?: any[];
}

export interface ChatResponse {
  response: string;
  sql?: string;
  data?: any[];
}

// Error Types
export interface ApiError {
  status: 'error';
  message: string;
  details?: string;
  code?: string;
}

export interface FileMetadata {
  name: string;
  size: number;
  type: string;
  last_modified: string;
  location: string;
}

export interface QualityMetrics {
  completeness: number;
  accuracy: number;
  consistency: number;
  details?: {
    total: number;
    valid: number;
    invalid: number;
    missing: number;
  };
}

export interface WorkflowInfo {
  id: string;
  name: string;
  status: string;
  created_at: string;
  updated_at: string;
  steps: WorkflowStep[];
}

export interface WorkflowStep {
  id: string;
  name: string;
  status: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
}

export interface QueryResult {
  columns: string[];
  rows: any[];
  total_rows: number;
  execution_time: number;
} 