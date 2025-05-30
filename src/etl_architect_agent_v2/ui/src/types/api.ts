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
  columns: ColumnInfo[];
  rowCount?: number;
  lastUpdated?: string;
  s3_location?: string;
  metadata?: Record<string, any>;
  created_at?: string;
  updated_at?: string;
}

export interface ColumnInfo {
  name: string;
  type: string;
  description?: string;
  sample_values?: any[];
  nullable?: boolean;
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
  file_name: string;
  s3_path: string;
  size: number;
  file_type: string;
  last_modified: string;
  table_name?: string;
}

export interface DataPreview {
  columns: string[];
  rows: any[][];
  totalRows: number;
}

export interface QualityMetrics {
  completeness: number;
  uniqueness: number;
  consistency: number;
  completeness_details?: {
    total: number;
    valid: number;
    invalid: number;
    missing: number;
  };
  uniqueness_details?: {
    total: number;
    valid: number;
    invalid: number;
    missing: number;
  };
  consistency_details?: {
    total: number;
    valid: number;
    invalid: number;
    missing: number;
  };
  metrics?: {
    [key: string]: {
      score: number;
      status: 'success' | 'warning' | 'error';
      details?: {
        total: number;
        valid: number;
        invalid: number;
        missing: number;
      };
    };
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

export interface ColumnSchema {
  name: string;
  type: string;
  comment: string;
  nullable: boolean;
}

export interface TableSchema {
  columns: ColumnSchema[];
  primaryKey?: string[];
  foreignKeys?: {
    column: string;
    references: {
      table: string;
      column: string;
    };
  }[];
}

export interface SavedQuery {
  query_id: string;
  name: string;
  description?: string;
  query: string;
  tables: string[];
  created_at: string;
  updated_at: string;
  created_by: string;
  is_favorite: boolean;
  last_run?: string;
  execution_count: number;
} 