const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  // Catalog endpoints
  CATALOG: `${API_BASE_URL}/api/catalog`,
  CATALOG_FILES: `${API_BASE_URL}/api/catalog/files`,
  CATALOG_TABLES: `${API_BASE_URL}/api/catalog/tables`,
  CATALOG_QUALITY: `${API_BASE_URL}/api/catalog/quality`,
  CATALOG_QUERY: `${API_BASE_URL}/api/catalog/query`,
  
  // File operations through catalog
  FILE_DETAILS: (s3Path: string) => `${API_BASE_URL}/api/catalog/files/${s3Path}/details`,
  FILE_PREVIEW: (s3Path: string) => `${API_BASE_URL}/api/catalog/files/${s3Path}/preview`,
  FILE_CONVERT: (s3Path: string) => `${API_BASE_URL}/api/catalog/files/${s3Path}/convert`,
  FILE_DELETE: (s3Path: string) => `${API_BASE_URL}/api/catalog/files/${s3Path}`,
  
  // Table operations through catalog
  TABLE_AUDIT: (tableName: string) => `${API_BASE_URL}/api/catalog/audit/${tableName}`,
  TABLE_QUALITY: (tableName: string) => `${API_BASE_URL}/api/catalog/quality/${tableName}`,
  TABLE_FILES: (tableName: string) => `${API_BASE_URL}/api/catalog/tables/${tableName}/files`,
}; 