import axios from 'axios';
import { FileInfo, TableInfo, QualityMetrics } from '../types/api';

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

export class CatalogService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  async getCatalog() {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog`);
      return response.data;
    } catch (error) {
      console.error('Error fetching catalog:', error);
      throw error;
    }
  }

  async listTables(userId: string = 'test_user'): Promise<TableInfo[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/tables`, {
        params: { user_id: userId }
      });
      return response.data;
    } catch (error) {
      console.error('Error listing tables:', error);
      throw error;
    }
  }

  async getTableDetails(tableName: string, userId: string = 'test_user'): Promise<TableInfo> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/tables/${tableName}`, {
        params: { user_id: userId }
      });
      return response.data;
    } catch (error) {
      console.error('Error getting table details:', error);
      throw error;
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
      console.error('Error getting table schema:', error);
      if (axios.isAxiosError(error)) {
        console.error('API Error details:', {
          status: error.response?.status,
          statusText: error.response?.statusText,
          data: error.response?.data
        });
      }
      throw error;
    }
  }

  async getTableQuality(tableName: string, userId: string = 'test_user'): Promise<QualityMetrics> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/catalog/tables/${tableName}/quality`, {
        params: { user_id: userId }
      });
      return response.data;
    } catch (error) {
      console.error('Error getting table quality:', error);
      throw error;
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
      console.error('Error uploading file:', error);
      throw error;
    }
  }

  async executeQuery(query: string, userId: string = 'test_user'): Promise<any> {
    try {
      const response = await axios.post(`${this.baseUrl}/api/catalog/query`, {
        query,
        user_id: userId
      });
      return response.data;
    } catch (error) {
      console.error('Error executing query:', error);
      throw error;
    }
  }

  async descriptiveQuery(
    query: string,
    tableName?: string,
    preserveColumnNames: string = "true",
    userId: string = "test_user"
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/catalog/descriptive_query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        table_name: tableName,
        preserve_column_names: preserveColumnNames,
        user_id: userId
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || "Failed to execute descriptive query");
    }

    return response.json();
  }

  async startWorkflow(workflowData: any) {
    try {
      const response = await axios.post(this.baseUrl + '/api/workflow/start', workflowData);
      return response.data;
    } catch (error) {
      console.error('Error starting workflow:', error);
      throw error;
    }
  }

  async listUserFiles(userId: string = 'test_user'): Promise<FileInfo[]> {
    try {
      const response = await axios.get(this.baseUrl + '/api/catalog/tables', {
        params: { user_id: userId }
      });
      
      // Transform the response to ensure status is always a string
      return response.data.map((file: any) => ({
        file_name: file.name || file.file_name,
        s3_path: file.location || file.s3_path,
        format: file.format,
        size: file.size,
        last_modified: file.last_modified,
        status: file.status || 'unknown' // Ensure status is always a string
      }));
    } catch (error) {
      console.error('Error listing user files:', error);
      throw error;
    }
  }

  async getFilePreview(s3Path: string, userId: string, rows: number = 5): Promise<any> {
    try {
      const response = await axios.get(this.baseUrl + '/api/files/' + s3Path + '/preview', {
        params: { 
          user_id: userId,
          rows: rows 
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error getting file preview:', error);
      throw error;
    }
  }

  async deleteFile(s3Path: string, userId: string): Promise<void> {
    try {
      await axios.delete(this.baseUrl + '/api/files/' + s3Path, {
        params: { user_id: userId }
      });
    } catch (error) {
      console.error('Error deleting file:', error);
      throw error;
    }
  }

  async getQualityMetrics(tableName: string): Promise<QualityMetrics> {
    try {
      const response = await axios.get(this.baseUrl + '/api/catalog/quality/' + tableName);
      return response.data;
    } catch (error) {
      console.error('Error getting quality metrics:', error);
      throw error;
    }
  }

  async getAvailableTransformations(): Promise<Array<{ value: string; label: string }>> {
    const response = await fetch('/api/transformation/types');
    if (!response.ok) {
      throw new Error('Failed to fetch transformation types');
    }
    const types = await response.json();
    return types.map((type: any) => ({
      value: type.type,
      label: type.name
    }));
  }

  async getTransformationTemplates(userId: string): Promise<TransformationTemplate[]> {
    const response = await fetch(`/api/transformation/templates?user_id=${userId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch transformation templates');
    }
    return response.json();
  }

  async saveTransformationTemplate(template: TransformationTemplate, userId: string): Promise<void> {
    const response = await fetch('/api/transformation/templates', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ template, user_id: userId })
    });
    if (!response.ok) {
      throw new Error('Failed to save transformation template');
    }
  }

  async deleteTransformationTemplate(templateName: string, userId: string): Promise<void> {
    const response = await fetch(`/api/transformation/templates/${templateName}?user_id=${userId}`, {
      method: 'DELETE'
    });
    if (!response.ok) {
      throw new Error('Failed to delete transformation template');
    }
  }

  async applyTransformation(
    tableName: string,
    config: TransformationConfig,
    userId: string
  ): Promise<TransformationResult> {
    const response = await fetch('/api/transformation/apply', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        table_name: tableName,
        config,
        user_id: userId
      })
    });
    if (!response.ok) {
      throw new Error('Failed to apply transformation');
    }
    return response.json();
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
}