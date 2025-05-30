import React, { useState, useEffect } from 'react';
import { getTables, runDescriptiveQuery, TableInfo, DescriptiveQueryResponse } from '../api';

type QueryHistoryItem = {
  query: string;
  timestamp: string;
};

const DataExplorer: React.FC = () => {
  const [tables, setTables] = useState<TableInfo[]>([]);
  const [selectedTable, setSelectedTable] = useState<string>('');
  const [query, setQuery] = useState<string>('');
  const [result, setResult] = useState<DescriptiveQueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [queryHistory, setQueryHistory] = useState<QueryHistoryItem[]>([]);

  useEffect(() => {
    const fetchTables = async () => {
      try {
        const data = await getTables();
        setTables(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to fetch tables');
      }
    };
    fetchTables();
  }, []);

  const handleRunQuery = async () => {
    if (!selectedTable || !query.trim()) {
      setError('Please select a table and enter a query');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const queryResult = await runDescriptiveQuery(query, selectedTable);
      setResult(queryResult);
      setQueryHistory(prev => [
        { query, timestamp: new Date().toLocaleString() },
        ...prev.slice(0, 9) // Keep last 10 queries
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Query execution failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleTableSelect = (tableName: string) => {
    setSelectedTable(tableName);
    // Auto-generate a simple descriptive query
    setQuery(`Show me all data from ${tableName}`);
  };

  return (
    <div style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div style={{ display: 'flex', gap: 16 }}>
        {/* Table Selector */}
        <div style={{ minWidth: 200 }}>
          <label style={{ display: 'block', marginBottom: 8, fontWeight: 500 }}>Select Table:</label>
          <select
            value={selectedTable}
            onChange={(e) => handleTableSelect(e.target.value)}
            style={{
              width: '100%',
              padding: 8,
              borderRadius: 4,
              border: '1px solid #ccc'
            }}
          >
            <option value="">Select a table...</option>
            {tables.map(t => (
              <option key={t.name} value={t.name}>{t.name}</option>
            ))}
          </select>
        </div>

        {/* Query Editor */}
        <div style={{ flex: 1 }}>
          <label style={{ display: 'block', marginBottom: 8, fontWeight: 500 }}>Natural Language Query:</label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{
              width: '100%',
              height: 100,
              padding: 8,
              fontFamily: 'monospace',
              borderRadius: 4,
              border: '1px solid #ccc'
            }}
            placeholder="Enter your query in natural language..."
          />
        </div>
      </div>

      {/* Run Button */}
      <div>
        <button
          onClick={handleRunQuery}
          disabled={loading || !selectedTable || !query.trim()}
          style={{
            padding: '8px 16px',
            background: '#1a73e8',
            color: 'white',
            border: 'none',
            borderRadius: 4,
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.7 : 1
          }}
        >
          {loading ? 'Running...' : 'Run Query'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{ color: 'red', padding: 12, background: '#ffebee', borderRadius: 4 }}>
          {error}
        </div>
      )}

      {/* Results Display */}
      {result?.results && result.results.length > 0 && (
        <div>
          <h3>Results</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 8 }}>
              <thead>
                <tr style={{ background: '#f0f2f7' }}>
                  {Object.keys(result.results[0] || {}).map(header => (
                    <th key={header} style={{ textAlign: 'left', padding: 8, border: '1px solid #ddd' }}>
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.results.map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((cell, j) => (
                      <td key={j} style={{ padding: 8, border: '1px solid #ddd' }}>
                        {String(cell)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Query History */}
      {queryHistory.length > 0 && (
        <div>
          <h3>Recent Queries</h3>
          <div style={{ background: '#f8f9fa', padding: 16, borderRadius: 4 }}>
            {queryHistory.map((item, i) => (
              <div key={i} style={{ marginBottom: 8 }}>
                <div style={{ fontFamily: 'monospace', marginBottom: 4 }}>{item.query}</div>
                <div style={{ fontSize: 12, color: '#666' }}>{item.timestamp}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DataExplorer; 