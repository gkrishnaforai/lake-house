import { TableInfo } from '../types/api';
import { TransformationTool, TransformationResult } from '../types/transformation';

export interface TableColumn {
    name: string;
    type: string;
    description: string;
}

export class TransformationService {
    private baseUrl: string;

    constructor() {
        this.baseUrl = '/api/transformation';
    }

    async getAvailableTransformations(): Promise<TransformationTool[]> {
        const response = await fetch(`${this.baseUrl}/tools`);
        if (!response.ok) {
            throw new Error('Failed to fetch transformation tools');
        }
        return response.json();
    }

    async getTableColumns(tableName: string, userId: string): Promise<TableColumn[]> {
        const response = await fetch(
            `${this.baseUrl}/tables/${tableName}/columns?user_id=${userId}`
        );
        if (!response.ok) {
            throw new Error('Failed to fetch table columns');
        }
        return response.json();
    }

    async applyTransformation(
        tableName: string,
        toolId: string,
        sourceColumns: string[],
        userId: string
    ): Promise<TransformationResult> {
        const response = await fetch(`${this.baseUrl}/apply`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                table_name: tableName,
                tool_id: toolId,
                source_columns: sourceColumns,
                user_id: userId,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || 'Failed to apply transformation');
        }

        return response.json();
    }

    async getTransformationTemplates(userId: string): Promise<any[]> {
        const response = await fetch(
            `${this.baseUrl}/templates?user_id=${userId}`
        );
        if (!response.ok) {
            throw new Error('Failed to fetch transformation templates');
        }
        return response.json();
    }

    async saveTransformationTemplate(
        template: any,
        userId: string
    ): Promise<any> {
        const response = await fetch(
            `${this.baseUrl}/templates?user_id=${userId}`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(template),
            }
        );
        if (!response.ok) {
            throw new Error('Failed to save transformation template');
        }
        return response.json();
    }

    async deleteTransformationTemplate(
        templateName: string,
        userId: string
    ): Promise<any> {
        const response = await fetch(
            `${this.baseUrl}/templates/${templateName}?user_id=${userId}`,
            {
                method: 'DELETE',
            }
        );
        if (!response.ok) {
            throw new Error('Failed to delete transformation template');
        }
        return response.json();
    }
}

export const transformationService = new TransformationService(); 