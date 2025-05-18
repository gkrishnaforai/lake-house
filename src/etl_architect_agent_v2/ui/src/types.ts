export interface CatalogData {
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  tables: TableInfo[];
  files: FileMetadata[];
  workflows: WorkflowState[];
}

export interface TableInfo {
  name: string;
  schema: any;
  location: string;
  description?: string;
  created_at: string;
  updated_at: string;
}

export interface FileMetadata {
  name: string;
  size: number;
  last_modified: string;
  format: string;
  location: string;
  schema?: any;
  quality?: QualityMetrics;
}

export interface QualityMetrics {
  completeness: number;
  accuracy: number;
  consistency: number;
  timeliness: number;
}

export interface WorkflowState {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed';
  currentStep: string;
  steps: WorkflowStep[];
  startTime: string;
  endTime?: string;
}

export interface WorkflowStep {
  id: string;
  name: string;
  status: 'completed' | 'in_progress' | 'pending' | 'error';
  progress: number;
  error?: string;
}

export interface AgentState {
  id: string;
  name: string;
  status: 'active' | 'inactive' | 'error' | 'unknown';
  lastSeen: string;
  currentTask?: string;
  error?: string;
}

// Component Props Interfaces
export interface CatalogDashboardProps {
  data: CatalogData | null;
  loading: boolean;
  error: string | null;
  onFileUpload: (file: File) => Promise<void>;
  onWorkflowStart: (workflowData: any) => Promise<void>;
  onErrorRecovery: (errorData: any) => Promise<void>;
  onRefresh: () => Promise<void>;
}

export interface AuditLogViewerProps {
  tableName: string;
  startDate?: Date;
  endDate?: Date;
  selectedTable?: string;
  selectedFile?: string;
}

export interface CatalogInterfaceProps {
  onFileUpload: (file: File) => Promise<void>;
  onQuery: (query: string) => Promise<void>;
  onTableSelect?: (table: TableInfo) => void;
  onFileSelect?: (file: FileMetadata) => void;
}

export interface DataQualityDashboardProps {
  tableName?: string;
  fileId?: string;
  selectedTable?: string;
  selectedFile?: string;
}

export interface FileExplorerProps {
  onFileSelect: (file: FileMetadata) => void;
  onFileDownload: (file: FileMetadata) => Promise<void>;
  onTableSelect?: (table: string | null) => void;
}

export interface FileUploadProps {
  onUpload: (file: File) => Promise<void>;
  onCancel: () => void;
  acceptedFormats?: string[];
  maxSize?: number;
  onUploadSuccess?: () => void;
  onConvertSuccess?: () => void;
}

export interface QueryInterfaceProps {
  onExecute: (query: string) => Promise<void>;
  onSave: (query: string, name: string) => Promise<void>;
  selectedTable?: string;
  selectedFile?: string;
} 