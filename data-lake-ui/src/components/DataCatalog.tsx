import React, { useEffect, useState } from 'react';
import { getTables, getTableFiles } from '../api';
import FileUpload from './FileUpload';

type TableInfo = {
  name: string;
  description?: string;
  columns: { name: string; type: string; description?: string }[];
  row_count?: number;
  last_updated?: string;
  s3_location?: string;
};

type FileMetadata = {
  file_name: string;
  file_type: string;
  s3_path: string;
  schema_path: string;
  created_at: string;
  size: number;
};

const DataCatalog: React.FC = () => {
  const [tables, setTables] = useState<TableInfo[]>([]);
  const [selected, setSelected] = useState<TableInfo | null>(null);
  const [files, setFiles] = useState<FileMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    getTables()
      .then(setTables)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selected) {
      setFiles([]);
      getTableFiles(selected.name)
        .then(setFiles)
        .catch(() => setFiles([]));
    }
  }, [selected]);

  return (
    <div style={{ display: 'flex', gap: 32 }}>
      <div style={{ minWidth: 260 }}>
        <h3>Tables</h3>
        {loading && <div>Loading...</div>}
        {error && <div style={{ color: 'red' }}>{error}</div>}
        <ul style={{ padding: 0, listStyle: 'none' }}>
          {tables.map(t => (
            <li key={t.name}>
              <button
                style={{
                  background: selected?.name === t.name ? '#e5e8ef' : 'none',
                  border: 'none',
                  padding: '8px 12px',
                  width: '100%',
                  textAlign: 'left',
                  cursor: 'pointer',
                  fontWeight: selected?.name === t.name ? 600 : 400,
                }}
                onClick={() => setSelected(t)}
              >
                {t.name}
              </button>
            </li>
          ))}
        </ul>
      </div>
      <div style={{ flex: 1 }}>
        {selected ? (
          <div>
            <h3>{selected.name}</h3>
            <div style={{ marginBottom: 16 }}>
              <b>Description:</b> {selected.description || 'N/A'}
            </div>
            <div style={{ marginBottom: 16 }}>
              <b>Schema:</b>
              <table style={{ width: '100%', marginTop: 8, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: '#f0f2f7' }}>
                    <th style={{ textAlign: 'left', padding: 4 }}>Name</th>
                    <th style={{ textAlign: 'left', padding: 4 }}>Type</th>
                    <th style={{ textAlign: 'left', padding: 4 }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {selected.columns.map(col => (
                    <tr key={col.name}>
                      <td style={{ padding: 4 }}>{col.name}</td>
                      <td style={{ padding: 4 }}>{col.type}</td>
                      <td style={{ padding: 4 }}>{col.description || ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ marginBottom: 16 }}>
              <b>Files:</b>
              <ul>
                {files.map(f => (
                  <li key={f.s3_path}>{f.file_name} ({f.file_type}, {Math.round(f.size/1024)} KB)</li>
                ))}
              </ul>
            </div>
            <FileUpload tableName={selected.name} onUploaded={() => getTableFiles(selected.name).then(setFiles)} />
          </div>
        ) : (
          <div>Select a table to view details and upload files.</div>
        )}
      </div>
    </div>
  );
};

export default DataCatalog; 