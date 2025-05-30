import { getUserId } from './authService';

// Base report interface for creating new reports
export interface ReportInput {
  name: string;
  description: string;
  query: string;
  schedule?: string;
  isFavorite?: boolean;
  charts?: Array<{
    id: string;
    type: string;
    title: string;
    xAxis: string;
    yAxis: string[];
    groupBy?: string;
    aggregation?: string;
  }>;
}

// Full report interface with all fields
export interface Report extends ReportInput {
  id: string;
  lastRun?: string;
  createdBy?: string;
  createdAt?: string;
  updatedAt?: string;
}

class ReportService {
  private baseUrl = '/api/reports';

  async getReports(): Promise<Report[]> {
    const userId = getUserId();
    const response = await fetch(`${this.baseUrl}/${userId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch reports');
    }
    return response.json();
  }

  async getReport(reportId: string): Promise<Report> {
    const userId = getUserId();
    const response = await fetch(`${this.baseUrl}/${userId}/${reportId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch report');
    }
    return response.json();
  }

  async saveReport(report: ReportInput): Promise<Report> {
    const userId = getUserId();
    const response = await fetch(`${this.baseUrl}/${userId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(report),
    });
    if (!response.ok) {
      throw new Error('Failed to save report');
    }
    return response.json();
  }

  async updateReport(reportId: string, report: ReportInput): Promise<Report> {
    const userId = getUserId();
    const response = await fetch(`${this.baseUrl}/${userId}/${reportId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(report),
    });
    if (!response.ok) {
      throw new Error('Failed to update report');
    }
    return response.json();
  }

  async deleteReport(reportId: string): Promise<void> {
    const userId = getUserId();
    const response = await fetch(`${this.baseUrl}/${userId}/${reportId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error('Failed to delete report');
    }
  }

  async runReport(reportId: string): Promise<any> {
    const userId = getUserId();
    const response = await fetch(`${this.baseUrl}/${userId}/${reportId}/run`, {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error('Failed to run report');
    }
    return response.json();
  }

  async scheduleReport(reportId: string, schedule: string): Promise<void> {
    const userId = getUserId();
    const response = await fetch(`${this.baseUrl}/${userId}/${reportId}/schedule`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ schedule }),
    });
    if (!response.ok) {
      throw new Error('Failed to schedule report');
    }
  }
}

export const reportService = new ReportService(); 