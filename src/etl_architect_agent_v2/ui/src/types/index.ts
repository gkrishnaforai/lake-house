import { Dispatch, SetStateAction } from 'react';

export interface FileExplorerProps {
  onFileSelect: Dispatch<SetStateAction<string | null>>;
  onTableSelect: Dispatch<SetStateAction<string | null>>;
}

export interface FileUploadProps {
  onUploadSuccess: () => void;
  onConvertSuccess: () => void;
}

export interface CatalogInterfaceProps {
  onTableSelect: Dispatch<SetStateAction<string | null>>;
  onFileSelect: Dispatch<SetStateAction<string | null>>;
}

export interface DataQualityDashboardProps {
  selectedTable: string | null;
  selectedFile: string | null;
}

export interface QueryInterfaceProps {
  selectedTable: string | null;
  selectedFile: string | null;
}

export interface SchemaViewerProps {
  selectedTable: string | null;
  selectedFile: string | null;
}

export interface AuditLogViewerProps {
  selectedTable: string | null;
  selectedFile: string | null;
}

export interface DataPreviewProps {
  filePath: string;
} 