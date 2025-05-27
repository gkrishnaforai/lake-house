import axios from 'axios';
import { FileInfo, TableInfo, QualityMetrics, FileMetadata } from '../types/api';

// Standardize base URL configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

export function getUserId(): string {
  // Always use 'test_user' to match backend default
  return 'test_user';
}

export interface TransformationConfig {
  source_columns: string[];
  transformation_type: string;
  new_column_name?: string;
  data_type?: string;
  parameters?: {
    categories?: string[];
    confidence_threshold?: number;
    aspects?: string[];
    language?: string;
  };
  is_template?: boolean;
  template_name?: string;
}

export interface TransformationTemplate {
  name: string;
  description: string;
  transformation_type: string;
  default_parameters: Record<string, any>;
  prompt_template?: string;
  example_input?: Record<string, any>;
  example_output?: Record<string, any>;
}

export interface TransformationResult {
  status: string;
  message?: string;
  new_columns: Array<{ name: string; type: string }>;
  preview_data?: Array<Record<string, any>>;
  errors?: string[];
}

export interface SavedQuery {
  id: string;
  name: string;
  query: string;
  isFavorite: boolean;
}

export interface ExecutionPlan {
  plan: string;
}

interface QueryResult {
  status: 'success' | 'error';
  results?: any[][];
  error?: string;
}

export class CatalogService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  private handleError(error: unknown, context: string): never {
    console.error(`Error in ${context}:`, error);
    if (axios.isAxiosError(error)) {
      const status = error.response?.status;
      const message = error.response?.data?.message || error.message;
      throw new Error(`API Error (${status}): ${message}`);
    }
    throw error instanceof Error ? error : new Error(`Unknown error in ${context}`);
  }

  async getCatalog() {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog`);
      return response.data;
    } catch (error) {
      this.handleError(error, 'getCatalog');
    }
  }

  async listTables(userId: string = 'test_user'): Promise<TableInfo[]> {
    try {
      console.log('Fetching tables for user:', userId);
      const response = await axios.get(`${this.baseUrl}/api/catalog/tables`, {
        params: { user_id: userId }
      });
      console.log('Raw API response:', response.data);
      
      // The response data is already an array, no need to access .tables
      const tables = Array.isArray(response.data) ? response.data : [];
      console.log('Tables array:', tables);
      
      const mappedTables = tables.map((table: any) => ({
        name: table.name,
        description: table.description || '',
        columns: table.schema ? Object.entries(table.schema).map(([name, type]: [string, any]) => ({
          name,
          type: typeof type === 'string' ? type : JSON.stringify(type),
          description: ''
        })) : [],
        rowCount: table.row_count || 0,
        lastUpdated: table.updated_at || table.created_at,
        s3_location: table.location || '',
        metadata: table.metadata || {},
        created_at: table.created_at,
        updated_at: table.updated_at
      }));
      
      console.log('Mapped tables:', mappedTables);
      return mappedTables;
    } catch (error) {
      console.error('Error in listTables:', error);
      this.handleError(error, 'listTables');
    }
  }

  async getTableDetails(tableName: string, userId: string = 'test_user'): Promise<TableInfo> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/tables/${tableName}`, {
        params: { user_id: userId }
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'getTableDetails');
    }
  }

  async getTableSchema(tableName: string, userId: string = 'test_user'): Promise<any> {
    try {
      console.log('Making schema request for table:', tableName);
      const response = await axios.get(`${this.baseUrl}/api/catalog/tables/${tableName}/schema`, {
        params: { user_id: userId }
      });
      console.log('Schema API response:', response.data);
      return response.data;
    } catch (error) {
      this.handleError(error, 'getTableSchema');
    }
  }

  async getTableQuality(tableName: string, userId: string = 'test_user'): Promise<QualityMetrics> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/tables/${tableName}/quality`, {
        params: { user_id: userId }
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'getTableQuality');
    }
  }

  async uploadFile(
    file: File, 
    tableName: string, 
    userId: string = 'test_user',
    createNew: boolean = false
  ): Promise<TableInfo> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('table_name', tableName);
      formData.append('user_id', userId);
      formData.append('create_new', String(createNew));

      const response = await axios.post(
        `${this.baseUrl}/api/catalog/tables/${tableName}/files`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response.data;
    } catch (error) {
      this.handleError(error, 'uploadFile');
    }
  }

  async executeQuery(query: string, userId: string): Promise<QueryResult> {
    try {
      const response = await axios.post(`${this.baseUrl}/api/catalog/query`, {
        query,
        user_id: userId,
      });

      return {
        status: 'success',
        results: response.data.results,
      };
    } catch (error) {
      if (axios.isAxiosError(error)) {
        return {
          status: 'error',
          error: error.response?.data?.message || error.message,
        };
      }
      return {
        status: 'error',
        error: error instanceof Error ? error.message : 'An unknown error occurred',
      };
    }
  }

  async descriptiveQuery(
    query: string,
    tableName?: string,
    preserveColumnNames: string = "true",
    userId: string = "test_user"
  ): Promise<any> {
    try {
      const response = await axios.post(`${this.baseUrl}/api/catalog/descriptive_query`, {
        query,
        table_name: tableName,
        preserve_column_names: preserveColumnNames,
        user_id: userId
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'descriptiveQuery');
    }
  }

  async startWorkflow(workflowData: any) {
    try {
      const response = await axios.post(`${this.baseUrl}/api/workflow/start`, workflowData);
      return response.data;
    } catch (error) {
      this.handleError(error, 'startWorkflow');
    }
  }

  async listUserFiles(userId: string = 'test_user'): Promise<FileInfo[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/tables`, {
        params: { user_id: userId }
      });
      
      return response.data.map((file: any) => ({
        file_name: file.name || file.file_name,
        s3_path: file.location || file.s3_path,
        format: file.format,
        size: file.size,
        last_modified: file.last_modified,
        status: file.status || 'unknown'
      }));
    } catch (error) {
      this.handleError(error, 'listUserFiles');
    }
  }

  async getFilePreview(s3Path: string, userId: string, rows: number = 5): Promise<any> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/files/preview`, {
        params: {
          s3_path: s3Path,
          user_id: userId,
          rows
        }
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'getFilePreview');
    }
  }

  async deleteFile(s3Path: string, userId: string): Promise<void> {
    try {
      await axios.delete(`${this.baseUrl}/api/catalog/files`, {
        params: {
          s3_path: s3Path,
          user_id: userId
        }
      });
    } catch (error) {
      this.handleError(error, 'deleteFile');
    }
  }

  async getQualityMetrics(tableName: string): Promise<QualityMetrics> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/tables/${tableName}/quality`);
      return response.data;
    } catch (error) {
      this.handleError(error, 'getQualityMetrics');
    }
  }

  async getAvailableTransformations(): Promise<Array<{ value: string; label: string }>> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/transformations`);
      return response.data;
    } catch (error) {
      this.handleError(error, 'getAvailableTransformations');
    }
  }

  async getTransformationTemplates(userId: string): Promise<TransformationTemplate[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/transformations/templates`, {
        params: { user_id: userId }
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'getTransformationTemplates');
    }
  }

  async saveTransformationTemplate(template: TransformationTemplate, userId: string): Promise<void> {
    try {
      await axios.post(`${this.baseUrl}/api/catalog/transformations/templates`, {
        ...template,
        user_id: userId
      });
    } catch (error) {
      this.handleError(error, 'saveTransformationTemplate');
    }
  }

  async deleteTransformationTemplate(templateName: string, userId: string): Promise<void> {
    try {
      await axios.delete(`${this.baseUrl}/api/catalog/transformations/templates/${templateName}`, {
        params: { user_id: userId }
      });
    } catch (error) {
      this.handleError(error, 'deleteTransformationTemplate');
    }
  }

  async applyTransformation(
    tableName: string,
    config: TransformationConfig,
    userId: string
  ): Promise<TransformationResult> {
    try {
      const response = await axios.post(`${this.baseUrl}/api/catalog/tables/${tableName}/transform`, {
        ...config,
        user_id: userId
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'applyTransformation');
    }
  }

  compareSchemas(oldSchema: any[], newSchema: any[]): Array<{
    type: 'added' | 'modified';
    column: string;
    details?: any;
    oldDetails?: any;
    newDetails?: any;
  }> {
    const changes: Array<{
      type: 'added' | 'modified';
      column: string;
      details?: any;
      oldDetails?: any;
      newDetails?: any;
    }> = [];
    
    const oldColumns = new Map(oldSchema.map(col => [col.name, col]));
    const newColumns = new Map(newSchema.map(col => [col.name, col]));

    // Find new columns
    newColumns.forEach((col, name) => {
      if (!oldColumns.has(name)) {
        changes.push({
          type: 'added',
          column: name,
          details: col
        });
      }
    });

    // Find modified columns
    oldColumns.forEach((oldCol, name) => {
      const newCol = newColumns.get(name);
      if (newCol && JSON.stringify(oldCol) !== JSON.stringify(newCol)) {
        changes.push({
          type: 'modified',
          column: name,
          oldDetails: oldCol,
          newDetails: newCol
        });
      }
    });

    return changes;
  }

  async getTableFiles(tableName: string): Promise<any[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/tables/${tableName}/files`);
      return response.data;
    } catch (error) {
      this.handleError(error, 'getTableFiles');
    }
  }

  async getTablePreview(
    tableName: string,
    options?: {
      page?: number;
      pageSize?: number;
      sortBy?: string;
      sortDirection?: 'asc' | 'desc';
      filters?: Record<string, string>;
    }
  ): Promise<any[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/tables/${tableName}/preview`, {
        params: options
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'getTablePreview');
    }
  }

  async getSavedQueries(): Promise<SavedQuery[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/queries`);
      return response.data;
    } catch (error) {
      this.handleError(error, 'getSavedQueries');
    }
  }

  async getQueryPlan(query: string): Promise<ExecutionPlan> {
    try {
      const response = await axios.post(`${this.baseUrl}/api/catalog/query/plan`, { query });
      return response.data;
    } catch (error) {
      this.handleError(error, 'getQueryPlan');
    }
  }

  async saveQuery(query: { name: string; query: string; isFavorite: boolean }): Promise<void> {
    try {
      await axios.post(`${this.baseUrl}/api/catalog/queries`, query);
    } catch (error) {
      this.handleError(error, 'saveQuery');
    }
  }

  async toggleQueryFavorite(queryId: string): Promise<void> {
    try {
      await axios.post(`${this.baseUrl}/api/catalog/queries/${queryId}/favorite`);
    } catch (error) {
      this.handleError(error, 'toggleQueryFavorite');
    }
  }

  async deleteQuery(queryId: string): Promise<void> {
    try {
      await axios.delete(`${this.baseUrl}/api/catalog/queries/${queryId}`);
    } catch (error) {
      this.handleError(error, 'deleteQuery');
    }
  }

  async getTransformationTools(): Promise<any[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/transformations/tools`);
      return response.data;
    } catch (error) {
      this.handleError(error, 'getTransformationTools');
    }
  }

  async sqlQuery(query: string, tables: string[], userId: string): Promise<any> {
    try {
      const response = await fetch('/api/catalog/query/sql', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          tables,
          user_id: userId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }

      return data.results;
    } catch (error) {
      console.error('Error executing SQL query:', error);
      throw error;
    }
  }

  async createReport(report: {
    name: string;
    description: string;
    query: string;
    schedule?: string;
  }, userId: string): Promise<any> {
    try {
      const response = await fetch('/api/catalog/reports', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...report,
          user_id: userId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error creating report:', error);
      throw error;
    }
  }

  async listReports(userId: string): Promise<any[]> {
    try {
      const response = await fetch(`/api/catalog/reports?user_id=${userId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error listing reports:', error);
      throw error;
    }
  }

  async updateReport(reportId: string, updates: {
    name?: string;
    description?: string;
    query?: string;
    schedule?: string;
    is_favorite?: boolean;
  }, userId: string): Promise<any> {
    try {
      const response = await fetch(`/api/catalog/reports/${reportId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...updates,
          user_id: userId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error updating report:', error);
      throw error;
    }
  }

  async deleteReport(reportId: string, userId: string): Promise<void> {
    try {
      const response = await fetch(`/api/catalog/reports/${reportId}?user_id=${userId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
    } catch (error) {
      console.error('Error deleting report:', error);
      throw error;
    }
  }

  async scheduleReport(reportId: string, schedule: string, userId: string): Promise<any> {
    try {
      const response = await fetch('/api/catalog/reports/schedule', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          report_id: reportId,
          schedule,
          user_id: userId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error scheduling report:', error);
      throw error;
    }
  }

  async shareReport(reportId: string, userId: string): Promise<any> {
    try {
      const response = await fetch('/api/catalog/reports/share', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          report_id: reportId,
          user_id: userId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error sharing report:', error);
      throw error;
    }
  }
}