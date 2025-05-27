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
  Stack,
  Tabs,
  Tab,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  ListItem,
  Divider,
  List
} from '@mui/material';
import {
  Search as SearchIcon,
  Download as DownloadIcon,
  FilterList as FilterIcon,
  Sort as SortIcon,
  Refresh as RefreshIcon,
  TableRows as TableIcon
} from '@mui/icons-material';
import { CatalogService } from '../services/catalogService';
import { TableInfo } from '../types/api';
import { TransformationTab } from './TransformationTab';
import TabPanel from './TabPanel';
import FileUpload from './FileUpload';

interface QueryResult {
  status: string;
  results?: any[][];
  query?: string;
  message?: string;
  columns_used?: string[];
  sql_query?: string;
  explanation?: string;
  confidence?: number;
  tables_used?: string[];
  filters?: Record<string, any>;
  error?: string | null;
}

interface DescriptiveQueryResult {
  status: string;
  results?: any[];
  query?: string;
  message?: string;
  columns_used?: string[];
  sql_query?: string;
  explanation?: string;
  confidence?: number;
  tables_used?: string[];
  filters?: Record<string, any>;
  error?: string | null;
}

interface QueryHistoryItem {
  query: string;
  sql: string;
  timestamp: string;
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
  const [queryHistory, setQueryHistory] = useState<QueryHistoryItem[]>([]);
  const [generatedSql, setGeneratedSql] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(0);
  const [selectedTable, setSelectedTable] = useState<TableInfo | null>(null);
  const [contextMenu, setContextMenu] = useState<{
    mouseX: number;
    mouseY: number;
    table: TableInfo | null;
  } | null>(null);
  const catalogService = new CatalogService();

  const handleTableClick = (table: TableInfo) => {
    console.log('Table clicked:', table.name);
    setSelectedTable(table);
    onTableSelect(table);
  };

  const handleContextMenuClose = () => {
    setContextMenu(null);
  };

  const handleContextMenu = (event: React.MouseEvent, table: TableInfo) => {
    event.preventDefault();
    setContextMenu({
      mouseX: event.clientX - 2,
      mouseY: event.clientY - 4,
      table
    });
  };

  const handleExport = async (table: TableInfo) => {
    try {
      const response = await fetch(`/api/catalog/tables/${table.name}/export`, {
        method: 'GET',
        headers: {
          'Accept': 'text/csv'
        }
      });
      
      if (!response.ok) {
        throw new Error('Export failed');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${table.name}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting table:', error);
      // You might want to show an error notification here
    }
    handleContextMenuClose();
  };

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
      setSelectedTable(table);  // Set the selectedTable state when selectedTables changes
      fetchSchema(table.name);
    } else {
      console.log('No tables selected, clearing schema');
      setSelectedTable(null);  // Clear selectedTable when no tables are selected
      setSchema([]);
      setShowSchemaAlert(false);
    }
  }, [selectedTables]);

  // Effect to monitor schema changes
  useEffect(() => {
    console.log('Schema state changed:', schema);
  }, [schema]);

  const handleQuerySubmit = async () => {
    console.log('Submitting query with selectedTable:', selectedTable);
    if (!selectedTable) {
      setError("Please select a table first");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await catalogService.descriptiveQuery(
        query,
        selectedTable.name,
        "true",
        "test_user"
      );
      
      if (result.status === "success") {
        console.log('Query result:', result);
        
        // If we have results but no columns, try to infer columns from the first row
        if (result.results && Array.isArray(result.results) && result.results.length > 0) {
          // Get column names from metadata if available
          if (result.metadata?.columns_used) {
            console.log('Using columns from metadata:', result.metadata.columns_used);
            setColumns(result.metadata.columns_used);
          } else if (result.columns && Array.isArray(result.columns)) {
            console.log('Using columns from result:', result.columns);
            setColumns(result.columns);
          } else {
            // Try to get column names from schema
            const schema = await catalogService.getTableSchema(selectedTable.name, "test_user");
            if (schema && schema.schema) {
              const columnNames = schema.schema.map((col: any) => col.name);
              console.log('Using columns from schema:', columnNames);
              setColumns(columnNames);
            } else {
              // Fallback to inferring from first row
              const firstRow = result.results[0];
              const inferredColumns = Object.keys(firstRow);
              console.log('Inferred columns from first row:', inferredColumns);
              setColumns(inferredColumns);
            }
          }
          
          // Process the results
          const dataObjects = result.results.map((row: any) => {
            if (typeof row === 'object' && row !== null) {
              return row; // Row is already an object
            } else if (Array.isArray(row)) {
              // Convert array to object using column names
              const obj: any = {};
              const headers = result.metadata?.columns_used || result.columns || 
                            Array.from({ length: row.length }, (_, i) => `Column ${i + 1}`);
              headers.forEach((col: string, i: number) => {
                obj[col] = row[i];
              });
              return obj;
            }
            return row;
          });
          
          console.log('Processed data objects:', dataObjects);
          setData(dataObjects);
          setGeneratedSql(result.query);
          setQueryHistory(prev => [...prev, {
            query,
            sql: result.query || '',
            timestamp: new Date().toISOString()
          }]);
        } else {
          setError('Info - No results returned from query');
        }
      } else {
        setError(result.message || "Query failed");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  // Add debug logging for data and columns state changes
  useEffect(() => {
    console.log('Data state updated:', data);
    console.log('Columns state updated:', columns);
    if (data.length > 0 && columns.length > 0) {
      //setError(`Info - Data grid should be visible with ${data.length} rows and ${columns.length} columns`);
    } else {
      setError(`Info - Data grid not visible. Data length: ${data.length}, Columns length: ${columns.length}`);
    }
  }, [data, columns]);

  // Add debug alert for table selection
  useEffect(() => {
    console.log('Selected tables changed:', selectedTables);
    if (selectedTables.length > 0) {
      setError(null); // Clear any existing error
    } else {
      setError('Info - No table selected');
    }
  }, [selectedTables]);

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleUploadSuccess = () => {
    console.log('Upload successful, refreshing schema...');
    // Refresh the table list or schema
    if (selectedTables.length > 0) {
      fetchSchema(selectedTables[0].name);
    }
  };

  const handleUploadError = (error: string) => {
    console.error('Upload error:', error);
    setError(error);
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={(_, newValue) => {
          console.log('Tab changed to:', newValue);
          setActiveTab(newValue);
        }}>
          <Tab label="Data" />
          <Tab label="Schema" />
          <Tab label="Transformations" />
          <Tab label="Upload" />
        </Tabs>
      </Box>

      <Menu
        open={contextMenu !== null}
        onClose={handleContextMenuClose}
        anchorReference="anchorPosition"
        anchorPosition={
          contextMenu !== null
            ? { top: contextMenu.mouseY, left: contextMenu.mouseX }
            : undefined
        }
      >
        <MenuItem onClick={() => contextMenu?.table && handleExport(contextMenu.table)}>
          <ListItemIcon>
            <DownloadIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Export as CSV</ListItemText>
        </MenuItem>
      </Menu>

      <TabPanel value={activeTab} index={0}>
        <Box sx={{ p: 2 }}>
          {selectedTables.length > 0 && (
            <Alert severity="info" sx={{ mb: 2 }}>
              Selected table: {selectedTables[0].name}
            </Alert>
          )}

          <Paper sx={{ p: 2, mb: 2 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Available Tables
                </Typography>
                <List>
                  {selectedTables.map((table) => (
                    <React.Fragment key={table.name}>
                      <ListItem 
                        button 
                        selected={selectedTable?.name === table.name}
                        onClick={() => handleTableClick(table)}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          setContextMenu({
                            mouseX: e.clientX - 2,
                            mouseY: e.clientY - 4,
                            table
                          });
                        }}
                      >
                        <ListItemIcon>
                          <TableIcon />
                        </ListItemIcon>
                        <ListItemText
                          primary={table.name}
                          secondary={table.description || 'No description'}
                        />
                        <Chip 
                          label={`${table.rowCount?.toLocaleString() || 'N/A'} rows`} 
                          color="primary" 
                          size="small" 
                        />
                      </ListItem>
                      <Divider />
                    </React.Fragment>
                  ))}
                </List>
              </Grid>
              <Grid item xs={12}>
                <ToggleButtonGroup
                  value={mode}
                  exclusive
                  onChange={handleModeChange}
                  size="small"
                  sx={{ mb: 2 }}
                >
                  <ToggleButton value="sql">SQL Query</ToggleButton>
                  <ToggleButton value="descriptive">Descriptive Query</ToggleButton>
                </ToggleButtonGroup>
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  multiline
                  rows={4}
                  variant="outlined"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={mode === 'sql' ? 'Enter your SQL query...' : 'Describe what you want to know about the data...'}
                  sx={{ mb: 2 }}
                />
              </Grid>
              <Grid item xs={12}>
                <Button
                  variant="contained"
                  onClick={handleQuerySubmit}
                  disabled={loading || !query.trim()}
                  startIcon={loading ? <CircularProgress size={20} /> : <SearchIcon />}
                >
                  {loading ? 'Running...' : 'Run Query'}
                </Button>
              </Grid>
            </Grid>
          </Paper>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {generatedSql && (
            <Paper sx={{ p: 2, mb: 2, bgcolor: 'grey.100' }}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Generated SQL:
              </Typography>
              <Typography
                component="pre"
                sx={{
                  p: 1,
                  bgcolor: 'white',
                  borderRadius: 1,
                  overflowX: 'auto',
                  fontSize: '0.875rem',
                  fontFamily: 'monospace'
                }}
              >
                {generatedSql}
              </Typography>
            </Paper>
          )}

          {data.length > 0 && columns.length > 0 && (
            <Paper sx={{ width: '100%', overflow: 'hidden' }}>
              <TableContainer sx={{ maxHeight: 440, overflowX: 'auto' }}>
                <Table stickyHeader>
                  <TableHead>
                    <TableRow>
                      {columns.map((column) => (
                        <TableCell key={column}>{column}</TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((row, rowIndex) => (
                      <TableRow
                        key={`row-${rowIndex}`}
                        sx={{
                          '&:nth-of-type(odd)': { bgcolor: 'grey.50' },
                          '&:hover': { bgcolor: 'grey.100' }
                        }}
                      >
                        {columns.map((column, colIndex) => (
                          <TableCell
                            key={`cell-${rowIndex}-${colIndex}`}
                            sx={{
                              fontSize: '0.875rem',
                              whiteSpace: 'nowrap',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              minWidth: 150
                            }}
                          >
                            {(() => {
                              // Handle array data
                              const value = Array.isArray(row) ? row[colIndex] : row[column];
                              if (value === null || value === undefined || value === '') {
                                return '-';
                              }
                              if (typeof value === 'object') {
                                return JSON.stringify(value);
                              }
                              return String(value);
                            })()}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              <TablePagination
                rowsPerPageOptions={[10, 25, 50, 100]}
                component="div"
                count={data.length}
                rowsPerPage={rowsPerPage}
                page={page}
                onPageChange={handleChangePage}
                onRowsPerPageChange={handleChangeRowsPerPage}
              />
            </Paper>
          )}
        </Box>
      </TabPanel>

      <TabPanel value={activeTab} index={1}>
        <Box sx={{ p: 2 }}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">
                      Schema for {selectedTables.length > 0 ? selectedTables[0].name : 'No table selected'}
                    </Typography>
                    <Button
                      size="small"
                      onClick={handleSchemaTabClick}
                      startIcon={<FilterIcon />}
                      variant={isSchemaTabActive ? "contained" : "outlined"}
                      disabled={selectedTables.length === 0}
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
                      {selectedTables.length === 0 ? 'Please select a table to view its schema.' : 'No schema information available for this table.'}
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
      </TabPanel>

      <TabPanel value={activeTab} index={2}>
        {selectedTables.length > 0 ? (
          <TransformationTab
            tableName={selectedTables[0].name}
            userId="test_user"
          />
        ) : (
          <Box sx={{ p: 2 }}>
            <Typography variant="body1" color="text.secondary">
              Please select a table to view transformation options.
            </Typography>
          </Box>
        )}
      </TabPanel>

      <TabPanel value={activeTab} index={3}>
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Upload File
          </Typography>
          <FileUpload
            onUploadSuccess={handleUploadSuccess}
            onError={handleUploadError}
            selectedTable={selectedTables.length > 0 ? selectedTables[0] : undefined}
          />
        </Box>
      </TabPanel>
    </Box>
  );
};

export default DataExplorer; 