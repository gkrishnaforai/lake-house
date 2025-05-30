import React, { useState, useEffect } from 'react';
import { getTables, getTableFiles } from '../api';

type TableInfo = {
  name: string;
  description?: string;
  columns: { name: string; type: string; description?: string }[];
  row_count?: number;
  last_updated?: string;
};

type FileMetadata = {
  file_name: string;
  file_type: string;
  s3_path: string;
  schema_path: string;
  created_at: string;
  size: number;
};

type Activity = {
  type: 'upload' | 'query';
  table: string;
  timestamp: string;
  details: string;
};

const Dashboard: React.FC = () => {
  const [tables, setTables] = useState<TableInfo[]>([]);
  const [recentFiles, setRecentFiles] = useState<FileMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activities, setActivities] = useState<Activity[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const tablesData = await getTables();
        setTables(tablesData);

        // Get files for each table
        const allFiles: FileMetadata[] = [];
        for (const table of tablesData) {
          const files = await getTableFiles(table.name);
          allFiles.push(...files);
        }
        setRecentFiles(allFiles.sort((a, b) => 
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        ).slice(0, 5));

        // Simulate some recent activities
        setActivities([
          {
            type: 'upload',
            table: 'sales_data',
            timestamp: new Date().toISOString(),
            details: 'Uploaded sales_data_2024.xlsx'
          },
          {
            type: 'query',
            table: 'customer_data',
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            details: 'SELECT * FROM customer_data LIMIT 100'
          }
        ]);
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const totalSize = recentFiles.reduce((sum, file) => sum + file.size, 0);
  const formatSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(2)} MB`;
  };

  if (loading) return <div style={{ padding: 24 }}>Loading dashboard...</div>;
  if (error) return <div style={{ padding: 24, color: 'red' }}>{error}</div>;

  return (
    <div style={{ padding: 24 }}>
      {/* Stats Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 24, marginBottom: 32 }}>
        <div style={{ background: 'white', padding: 24, borderRadius: 8, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
          <h3 style={{ margin: 0, color: '#666' }}>Total Tables</h3>
          <div style={{ fontSize: 32, fontWeight: 'bold', marginTop: 8 }}>{tables.length}</div>
        </div>
        <div style={{ background: 'white', padding: 24, borderRadius: 8, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
          <h3 style={{ margin: 0, color: '#666' }}>Total Files</h3>
          <div style={{ fontSize: 32, fontWeight: 'bold', marginTop: 8 }}>{recentFiles.length}</div>
        </div>
        <div style={{ background: 'white', padding: 24, borderRadius: 8, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
          <h3 style={{ margin: 0, color: '#666' }}>Total Size</h3>
          <div style={{ fontSize: 32, fontWeight: 'bold', marginTop: 8 }}>{formatSize(totalSize)}</div>
        </div>
      </div>

      {/* Recent Activity */}
      <div style={{ background: 'white', padding: 24, borderRadius: 8, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
        <h2 style={{ margin: 0, marginBottom: 16 }}>Recent Activity</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {activities.map((activity, i) => (
            <div key={i} style={{ 
              padding: 12, 
              background: '#f8f9fa', 
              borderRadius: 4,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <div>
                <div style={{ fontWeight: 500 }}>
                  {activity.type === 'upload' ? '📤 File Upload' : '🔍 Query Executed'}
                </div>
                <div style={{ color: '#666', fontSize: 14 }}>
                  Table: {activity.table}
                </div>
                <div style={{ color: '#666', fontSize: 14 }}>
                  {activity.details}
                </div>
              </div>
              <div style={{ color: '#666', fontSize: 14 }}>
                {new Date(activity.timestamp).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div style={{ marginTop: 32 }}>
        <h2 style={{ margin: 0, marginBottom: 16 }}>Quick Actions</h2>
        <div style={{ display: 'flex', gap: 16 }}>
          <button
            onClick={() => window.location.href = '#/catalog'}
            style={{
              padding: '12px 24px',
              background: '#1a73e8',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              cursor: 'pointer'
            }}
          >
            Upload New File
          </button>
          <button
            onClick={() => window.location.href = '#/explorer'}
            style={{
              padding: '12px 24px',
              background: '#34a853',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              cursor: 'pointer'
            }}
          >
            Run Query
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 