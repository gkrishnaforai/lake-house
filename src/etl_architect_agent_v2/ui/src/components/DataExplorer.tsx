import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Chip,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  CardActions,
  IconButton,
  Tooltip,
  ToggleButton,
  ToggleButtonGroup,
  Stack
} from '@mui/material';
import {
  Search as SearchIcon,
  Download as DownloadIcon,
  FilterList as FilterIcon,
  Sort as SortIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { CatalogService } from '../services/catalogService';
import { TableInfo } from '../types/api';

interface QueryResult {
  status: string;
  results?: any[][];
  query?: string;
  message?: string;
}

interface DescriptiveQueryResult {
  status: string;
  results?: any[];
  query?: string;
  message?: string;
}

interface DataExplorerProps {
  selectedTables: TableInfo[];
  onTableSelect: (table: TableInfo) => void;
}

const DataExplorer: React.FC<DataExplorerProps> = ({ selectedTables, onTableSelect }) => {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'sql' | 'descriptive'>('sql');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [data, setData] = useState<any[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [schema, setSchema] = useState<{ name: string; type: string; comment: string }[]>([]);
  const [schemaCache, setSchemaCache] = useState<Record<string, { name: string; type: string; comment: string }[]>>({});
  const [isSchemaTabActive, setIsSchemaTabActive] = useState(false);
  const [showSchemaAlert, setShowSchemaAlert] = useState(false);
  const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const [generatedSql, setGeneratedSql] = useState<string | null>(null);
  const catalogService = new CatalogService();

  // Add a separate function to handle schema fetching
  const fetchSchema = async (tableName: string) => {
    try {
      // Check if schema is already in cache
      if (schemaCache[tableName]) {
        console.log('Using cached schema for table:', tableName);
        setSchema(schemaCache[tableName]);
        setShowSchemaAlert(true);
        return;
      }

      console.log('Fetching schema for table:', tableName);
      const data = await catalogService.getTableSchema(tableName);
      console.log('Raw schema data received:', data);
      
      if (data && Array.isArray(data.schema)) {
        const mappedSchema = data.schema.map((col: any) => ({
          name: col.name,
          type: col.type,
          comment: col.comment
        }));
        console.log('Mapped schema before setState:', mappedSchema);
        
        // Update both schema state and cache
        setSchema(mappedSchema);
        setSchemaCache(prev => ({
          ...prev,
          [tableName]: mappedSchema
        }));
        setShowSchemaAlert(true);
        console.log('Schema state and cache updated');
      } else {
        console.log('Invalid schema data format:', data);
        setSchema([]);
        setShowSchemaAlert(false);
      }
    } catch (error) {
      console.error('Error in fetchSchema:', error);
      setSchema([]);
      setShowSchemaAlert(false);
    }
  };

  // Handle mode change for query types
  const handleModeChange = (_: any, newMode: 'sql' | 'descriptive') => {
    if (newMode) setMode(newMode);
    setQuery('');
    setData([]);
    setColumns([]);
    setError(null);
    setDownloadUrl(null);
  };

  // Handle schema tab click
  const handleSchemaTabClick = () => {
    console.log('Schema tab clicked');
    setIsSchemaTabActive(true);
    if (selectedTables.length > 0) {
      const table = selectedTables[0];
      console.log('Fetching schema for table:', table.name);
      fetchSchema(table.name);
    }
  };

  // Effect for table selection
  useEffect(() => {
    console.log('useEffect triggered with selectedTables:', selectedTables);
    if (selectedTables.length > 0) {
      const table = selectedTables[0];
      console.log('Table selected:', table.name);
      fetchSchema(table.name);
    } else {
      console.log('No tables selected, clearing schema');
      setSchema([]);
      setShowSchemaAlert(false);
    }
  }, [selectedTables]);

  // Effect to monitor schema changes
  useEffect(() => {
    console.log('Schema state changed:', schema);
  }, [schema]);

  const handleQuerySubmit = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setData([]);
    setColumns([]);
    setDownloadUrl(null);
    setGeneratedSql(null);
    setQueryHistory((prev) => [query, ...prev.filter((q) => q !== query)].slice(0, 10));
    try {
      let result: QueryResult | DescriptiveQueryResult;
      if (mode === 'sql') {
        const response = await fetch('/api/catalog/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        });
        if (!response.ok) throw new Error('Query failed');
        result = await response.json();
        console.log('SQL Query Result:', result);
        if (Array.isArray(result.results)) {
          if (result.results.length > 0) {
            if (Array.isArray(result.results[0])) {
              console.log('Results is an array, length:', result.results.length);
              console.log('First result item:', result.results[0]);
       
              // Handle array of arrays format
              setColumns(result.results[0]);
              const dataObjects = result.results.slice(1).map((row: any[]) => {
                const obj: any = {};
                if (result.results && result.results[0]) {
                  result.results[0].forEach((col: string, i: number) => {
                    obj[col] = row[i];
                  });
                }
                return obj;
              });
              setData(dataObjects);
            } else {
              // Handle array of objects format
              setColumns(Object.keys(result.results[0]));
              setData(result.results);
            }
          }
        }
      } else {
        // Descriptive query
        const resp = await catalogService.descriptiveQuery(
          query, 
          selectedTables.length > 0 ? selectedTables[0].name : undefined,
          "true",
          "test_user"
        );
        console.log('Generated SQL Query:', resp.sql_query);

        if (resp.status === 'success') {
          console.log('Generated SQL Query:', resp.query);
          
          setData(resp.results || []);
          // If results is a list of lists (from Athena), use the first row as columns
          if (resp.results && resp.results.length > 0) {
            if (Array.isArray(resp.results[0])) {
              setColumns(resp.results[0] || []);
              // Convert list of lists to list of objects for display
              const dataObjects = resp.results.slice(1).map((row: any[]) => {
                const obj: any = {};
                if (resp.results && resp.results[0]) {
                  resp.results[0].forEach((col: string, i: number) => {
                    obj[col] = row[i];
                  });
                }
                return obj;
              });
              setData(dataObjects);
            } else {
              setColumns(Object.keys(resp.results[0] || {}));
            }
          }
          setGeneratedSql(resp.query || null);
        } else {
          setError(resp.message || 'Query failed');
        }
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Query failed');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    if (!data.length) return;
    // Export as CSV
    const csvRows = [columns.join(',')];
    for (const row of data) {
      csvRows.push(columns.map(col => JSON.stringify(row[col] ?? '')).join(','));
    }
    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    setDownloadUrl(url);
    setTimeout(() => window.URL.revokeObjectURL(url), 10000);
  };

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Explore Your Data
              </Typography>
              <ToggleButtonGroup
                value={mode}
                exclusive
                onChange={handleModeChange}
                sx={{ mb: 2 }}
              >
                <ToggleButton value="sql">SQL</ToggleButton>
                <ToggleButton value="descriptive">Descriptive</ToggleButton>
              </ToggleButtonGroup>
              <Typography variant="body2" color="text.secondary" paragraph>
                {mode === 'sql'
                  ? 'Write SQL queries to explore your data.'
                  : 'Ask questions in natural language. (Table selection is optional)'}
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                {mode === 'sql' ? (
                  <TextField
                    fullWidth
                    multiline
                    rows={4}
                    variant="outlined"
                    placeholder="SELECT * FROM your_table LIMIT 10"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                ) : (
                  <TextField
                    fullWidth
                    variant="outlined"
                    placeholder="e.g. Show me all staff in the Engineering department"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                )}
              </Box>
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                {queryHistory.length > 0 && (
                  <Chip
                    label={queryHistory[0].length > 40 ? queryHistory[0].slice(0, 37) + '...' : queryHistory[0]}
                    onClick={() => setQuery(queryHistory[0])}
                    size="small"
                    variant="outlined"
                  />
                )}
                <Button
                  variant="contained"
                  startIcon={<SearchIcon />}
                  onClick={handleQuerySubmit}
                  disabled={loading || !query.trim()}
                >
                  Run Query
                </Button>
                {data.length > 0 && (
                  downloadUrl ? (
                    <a
                      href={downloadUrl}
                      download="query_results.csv"
                      style={{ textDecoration: 'none' }}
                    >
                      <Button
                        variant="outlined"
                        startIcon={<DownloadIcon />}
                        disabled={loading}
                      >
                        Download CSV
                      </Button>
                    </a>
                  ) : (
                    <Button
                      variant="outlined"
                      startIcon={<DownloadIcon />}
                      onClick={handleExport}
                      disabled={loading || data.length === 0}
                    >
                      Download CSV
                    </Button>
                  )
                )}
                <Tooltip title="Refresh">
                  <IconButton onClick={handleQuerySubmit} disabled={loading}>
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {error && (
          <Grid item xs={12}>
            <Alert severity="error">{error}</Alert>
          </Grid>
        )}

        {loading && (
          <Grid item xs={12} sx={{ textAlign: 'center', py: 4 }}>
            <CircularProgress />
          </Grid>
        )}

        {data.length > 0 && (
          <Grid item xs={12}>
            {generatedSql && (
              <Paper sx={{ p: 2, mb: 2, bgcolor: 'grey.50' }}>
                <Typography variant="subtitle2" gutterBottom>
                  Generated SQL Query:
                </Typography>
                <Box
                  component="pre"
                  sx={{
                    p: 2,
                    bgcolor: 'grey.100',
                    borderRadius: 1,
                    overflowX: 'auto',
                    fontFamily: 'monospace',
                    fontSize: '0.875rem',
                  }}
                >
                  {generatedSql}
                </Box>
              </Paper>
            )}
            <Paper sx={{ width: '100%', overflow: 'hidden' }}>
              <TableContainer sx={{ maxHeight: 440 }}>
                <Table stickyHeader>
                  <TableHead>
                    <TableRow>
                      {columns.map((column) => (
                        <TableCell key={column}>
                          {column}
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data
                      .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                      .map((row, rowIndex) => (
                        <TableRow key={rowIndex}>
                          {columns.map((column) => (
                            <TableCell key={column}>
                              {typeof row[column] === 'boolean' ? (
                                <Chip
                                  label={row[column] ? 'Yes' : 'No'}
                                  color={row[column] ? 'success' : 'error'}
                                  size="small"
                                />
                              ) : (
                                row[column]
                              )}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </TableContainer>
              <TablePagination
                rowsPerPageOptions={[10, 25, 100]}
                component="div"
                count={data.length}
                rowsPerPage={rowsPerPage}
                page={page}
                onPageChange={handleChangePage}
                onRowsPerPageChange={handleChangeRowsPerPage}
              />
            </Paper>
          </Grid>
        )}
      </Grid>

      {selectedTables.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Schema for {selectedTables[0].name}
                </Typography>
                <Button
                  size="small"
                  onClick={handleSchemaTabClick}
                  startIcon={<FilterIcon />}
                  variant={isSchemaTabActive ? "contained" : "outlined"}
                >
                  {isSchemaTabActive ? "Hide Schema" : "View Schema"}
                </Button>
              </Box>
              {showSchemaAlert && schema.length > 0 && (
                <Alert 
                  severity="info" 
                  onClose={() => setShowSchemaAlert(false)}
                  sx={{ mb: 2 }}
                >
                  Found {schema.length} columns in the schema.
                </Alert>
              )}
              {isSchemaTabActive && schema.length > 0 ? (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell><strong>Column Name</strong></TableCell>
                        <TableCell><strong>Type</strong></TableCell>
                        <TableCell><strong>Description</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {schema.map((col) => (
                        <TableRow key={col.name}>
                          <TableCell>{col.name}</TableCell>
                          <TableCell>{col.type}</TableCell>
                          <TableCell>{col.comment}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : schema.length > 0 ? (
                <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', gap: 1 }}>
                  {schema.map((col) => (
                    <Chip 
                      key={col.name} 
                      label={`${col.name}: ${col.type}`} 
                      size="small"
                      title={col.comment}
                    />
                  ))}
                </Stack>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No schema information available for this table.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
};

export default DataExplorer; 