import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Paper,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  CircularProgress,
  Alert,
  Drawer,
  List,
  ListItem,
  ListItemText,
  TextField,
  Chip,
  IconButton,
  Tooltip,
  Divider,
  Badge,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CardHeader
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  TableChart as TableIcon,
  Assessment as QualityIcon,
  Schema as SchemaIcon,
  Search as SearchIcon,
  Info as InfoIcon,
  Add as AddIcon,
  Remove as RemoveIcon,
  Refresh as RefreshIcon,
  MoreVert as MoreVertIcon,
  Preview as PreviewIcon,
  InsertDriveFile as FileIcon,
  Description as DescriptionIcon,
  Storage as StorageIcon,
  Transform as TransformIcon,
  Close as CloseIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import FileUpload from './FileUpload';
import DataExplorer from './DataExplorer';
import QualityMetrics from './QualityMetrics';
import SchemaExplorer from './SchemaExplorer';
import { CatalogService } from '../services/catalogService';
import { FileMetadata, TableInfo } from '../types/api';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div role="tabpanel" hidden={value !== index}>
    {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
  </div>
);

const drawerWidth = 300;

const SchemaDialog: React.FC<{
  open: boolean;
  onClose: () => void;
  tableName: string;
  fetchSchema: (tableName: string) => Promise<any[]>;
}> = ({ open, onClose, tableName, fetchSchema }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [transformationTools, setTransformationTools] = useState<any[]>([]);
  const [schema, setSchema] = useState<any[]>([]);
  const catalogService = new CatalogService();

  useEffect(() => {
    const fetchData = async () => {
      if (!open || !tableName) return;

      setLoading(true);
      setError(null);

      try {
        const [toolsData, schemaData] = await Promise.all([
          catalogService.getTransformationTools(),
          fetchSchema(tableName)
        ]);
        setTransformationTools(toolsData);
        setSchema(schemaData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [open, tableName, fetchSchema]);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          minHeight: '80vh',
          maxHeight: '90vh',
          bgcolor: '#f5f5f5'
        }
      }}
    >
      <DialogTitle sx={{ 
        bgcolor: 'primary.main', 
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        py: 2
      }}>
        <Box display="flex" alignItems="center">
          <SchemaIcon sx={{ mr: 1 }} />
          <Typography variant="h6">
            Schema Explorer - {tableName}
          </Typography>
        </Box>
        <IconButton onClick={onClose} size="small" sx={{ color: 'white' }}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers sx={{ p: 3 }}>
        {loading ? (
          <Box display="flex" justifyContent="center" p={3}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        ) : (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              {schema.length > 0 ? (
                <Card elevation={2} sx={{ 
                  borderRadius: 2,
                  overflow: 'hidden'
                }}>
                  <CardHeader
                    title="Table Schema"
                    avatar={<SchemaIcon color="primary" />}
                    sx={{
                      bgcolor: 'white',
                      borderBottom: '1px solid',
                      borderColor: 'divider'
                    }}
                  />
                  <CardContent sx={{ p: 0 }}>
                    <TableContainer sx={{ 
                      maxHeight: '60vh',
                      overflow: 'auto',
                      '& .MuiTable-root': {
                        minWidth: 650
                      }
                    }}>
                      <Table size="small" stickyHeader>
                        <TableHead>
                          <TableRow sx={{ bgcolor: 'grey.100' }}>
                            <TableCell sx={{ 
                              fontWeight: 'bold',
                              position: 'sticky',
                              left: 0,
                              bgcolor: 'grey.100',
                              zIndex: 1
                            }}>Column Name</TableCell>
                            <TableCell sx={{ 
                              fontWeight: 'bold',
                              position: 'sticky',
                              left: 200,
                              bgcolor: 'grey.100',
                              zIndex: 1
                            }}>Data Type</TableCell>
                            <TableCell sx={{ fontWeight: 'bold' }}>Description</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {schema.map((column, index) => (
                            <TableRow 
                              key={column.name}
                              sx={{ 
                                '&:nth-of-type(odd)': { bgcolor: 'white' },
                                '&:nth-of-type(even)': { bgcolor: 'grey.50' },
                                '&:hover': { bgcolor: 'grey.100' }
                              }}
                            >
                              <TableCell sx={{ 
                                fontWeight: 'medium',
                                position: 'sticky',
                                left: 0,
                                bgcolor: 'inherit',
                                zIndex: 1
                              }}>{column.name}</TableCell>
                              <TableCell sx={{ 
                                position: 'sticky',
                                left: 200,
                                bgcolor: 'inherit',
                                zIndex: 1
                              }}>
                                <Chip 
                                  label={column.type} 
                                  size="small" 
                                  color="primary" 
                                  variant="outlined"
                                />
                              </TableCell>
                              <TableCell>{column.comment || '-'}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              ) : (
                <Alert severity="info">No schema information available</Alert>
              )}
            </Grid>

            {transformationTools && transformationTools.length > 0 && (
              <Grid item xs={12}>
                <Card elevation={2} sx={{ 
                  borderRadius: 2,
                  overflow: 'hidden'
                }}>
                  <CardHeader
                    title="Available Transformations"
                    avatar={<TransformIcon color="primary" />}
                    sx={{
                      bgcolor: 'white',
                      borderBottom: '1px solid',
                      borderColor: 'divider'
                    }}
                  />
                  <CardContent>
                    <Grid container spacing={2}>
                      {transformationTools.map((tool) => (
                        <Grid item xs={12} key={tool.id}>
                          <Card 
                            variant="outlined" 
                            sx={{ 
                              borderRadius: 2,
                              '&:hover': {
                                boxShadow: 2,
                                bgcolor: 'grey.50'
                              }
                            }}
                          >
                            <CardContent>
                              <Typography variant="h6" gutterBottom color="primary">
                                {tool.name}
                              </Typography>
                              <Typography variant="body2" color="text.secondary" paragraph>
                                {tool.description}
                              </Typography>
                              {tool.example_input && (
                                <Box mt={2}>
                                  <Typography variant="subtitle2" gutterBottom color="primary">
                                    Example Input:
                                  </Typography>
                                  <Typography 
                                    variant="body2" 
                                    component="pre" 
                                    sx={{ 
                                      bgcolor: 'grey.100', 
                                      p: 2, 
                                      borderRadius: 1,
                                      border: '1px solid',
                                      borderColor: 'divider'
                                    }}
                                  >
                                    {JSON.stringify(tool.example_input, null, 2)}
                                  </Typography>
                                </Box>
                              )}
                              {tool.example_output && (
                                <Box mt={2}>
                                  <Typography variant="subtitle2" gutterBottom color="primary">
                                    Example Output:
                                  </Typography>
                                  <Typography 
                                    variant="body2" 
                                    component="pre" 
                                    sx={{ 
                                      bgcolor: 'grey.100', 
                                      p: 2, 
                                      borderRadius: 1,
                                      border: '1px solid',
                                      borderColor: 'divider'
                                    }}
                                  >
                                    {JSON.stringify(tool.example_output, null, 2)}
                                  </Typography>
                                </Box>
                              )}
                            </CardContent>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            )}
          </Grid>
        )}
      </DialogContent>
      <DialogActions sx={{ bgcolor: 'white', p: 2 }}>
        <Button 
          onClick={onClose}
          variant="contained"
          color="primary"
          startIcon={<CloseIcon />}
        >
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const DataPreviewDialog: React.FC<{
  open: boolean;
  onClose: () => void;
  table: TableInfo | null;
}> = ({ open, onClose, table }) => {
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const catalogService = new CatalogService();

  useEffect(() => {
    if (open && table) {
      setLoading(true);
      setError(null);
      catalogService.getTablePreview(table.name)
        .then(data => {
          setPreviewData(Array.isArray(data) ? data : []);
        })
        .catch(err => {
          setError(err.message || 'Failed to load preview data');
        })
        .finally(() => {
          setLoading(false);
        });
    }
  }, [open, table]);

  if (!table) return null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box display="flex" alignItems="center">
          <PreviewIcon sx={{ mr: 1 }} />
          Data Preview for {table.name}
        </Box>
      </DialogTitle>
      <DialogContent>
        {loading ? (
          <Box display="flex" justifyContent="center" p={3}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Alert severity="error">{error}</Alert>
        ) : previewData.length > 0 ? (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  {table.columns.map((column) => (
                    <TableCell key={column.name}>{column.name}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {previewData.map((row, index) => (
                  <TableRow key={index}>
                    {table.columns.map((column) => (
                      <TableCell key={column.name}>{row[column.name] || '-'}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Alert severity="info">No preview data available</Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

const TableFilesDialog: React.FC<{
  open: boolean;
  onClose: () => void;
  table: TableInfo | null;
}> = ({ open, onClose, table }) => {
  const [files, setFiles] = useState<FileMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const catalogService = new CatalogService();

  useEffect(() => {
    const fetchFiles = async () => {
      if (!open || !table) return;

      setLoading(true);
      setError(null);

      try {
        const data = await catalogService.getTableFiles(table.name);
        setFiles(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch files');
      } finally {
        setLoading(false);
      }
    };

    fetchFiles();
  }, [open, table]);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          minHeight: '80vh',
          maxHeight: '90vh',
          bgcolor: '#f5f5f5'
        }
      }}
    >
      <DialogTitle sx={{ 
        bgcolor: 'primary.main', 
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        py: 2
      }}>
        <Box display="flex" alignItems="center">
          <FileIcon sx={{ mr: 1 }} />
          <Typography variant="h6">
            Table Files - {table?.name}
          </Typography>
        </Box>
        <IconButton onClick={onClose} size="small" sx={{ color: 'white' }}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers sx={{ p: 3 }}>
        {loading ? (
          <Box display="flex" justifyContent="center" p={3}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        ) : files.length > 0 ? (
          <TableContainer sx={{ 
            maxHeight: '60vh',
            overflow: 'auto',
            '& .MuiTable-root': {
              minWidth: 650
            }
          }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow sx={{ bgcolor: 'grey.100' }}>
                  <TableCell sx={{ fontWeight: 'bold' }}>File Name</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Format</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Size</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Last Modified</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Location</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {files.map((file) => (
                  <TableRow key={file.file_name}>
                    <TableCell>{file.file_name}</TableCell>
                    <TableCell>{file.file_type}</TableCell>
                    <TableCell>{(file.size / 1024).toFixed(2)} KB</TableCell>
                    <TableCell>{new Date(file.last_modified).toLocaleString()}</TableCell>
                    <TableCell>
                      <Tooltip title={file.s3_path}>
                        <Typography noWrap sx={{ maxWidth: 300 }}>
                          {file.s3_path}
                        </Typography>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Alert severity="info">No files found for this table</Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

const CatalogDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedTables, setSelectedTables] = useState<TableInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tables, setTables] = useState<TableInfo[]>([]);
  const [filteredTables, setFilteredTables] = useState<TableInfo[]>([]);
  const [search, setSearch] = useState('');
  const [userId, setUserId] = useState('test_user');
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedTableForMenu, setSelectedTableForMenu] = useState<TableInfo | null>(null);
  const [schemaDialogOpen, setSchemaDialogOpen] = useState(false);
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false);
  const [tableFileCounts, setTableFileCounts] = useState<Record<string, number>>({});
  const [selectedTable, setSelectedTable] = useState<TableInfo | null>(null);
  const [showSchema, setShowSchema] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [schemaCache, setSchemaCache] = useState<Record<string, any[]>>({});
  const [showFiles, setShowFiles] = useState(false);
  const catalogService = new CatalogService();

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleUploadSuccess = () => {
    setActiveTab(1);
    fetchTables();
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
  };

  const fetchTables = async () => {
    console.log('Starting to fetch tables...');
    setLoading(true);
    try {
      console.log('Calling catalogService.listTables...');
      const data = await catalogService.listTables(userId);
      console.log('Received tables data:', data);
      setTables(data);
      setFilteredTables(data);
      console.log('Updated tables state:', data);
    } catch (err) {
      console.error('Error in fetchTables:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch tables');
    } finally {
      setLoading(false);
      console.log('Finished fetching tables');
    }
  };

  useEffect(() => {
    console.log('Initial useEffect triggered, userId:', userId);
    fetchTables();
  }, [userId]);

  useEffect(() => {
    console.log('Search/filter effect triggered');
    console.log('Current search:', search);
    console.log('Current tables:', tables);
    
    if (!search) {
      console.log('No search term, setting all tables');
      setFilteredTables(tables);
    } else {
      const filtered = tables.filter((t) => 
        t.name.toLowerCase().includes(search.toLowerCase()) ||
        t.description?.toLowerCase().includes(search.toLowerCase())
      );
      console.log('Filtered tables:', filtered);
      setFilteredTables(filtered);
    }
  }, [search, tables]);

  const handleTableSelect = (table: TableInfo) => {
    setSelectedTables(prev => {
      const exists = prev.some(t => t.name === table.name);
      if (exists) {
        return prev.filter(t => t.name !== table.name);
      }
      return [...prev, table];
    });
  };

  const handleTableRemove = (tableName: string) => {
    setSelectedTables(prev => prev.filter(t => t.name !== tableName));
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, table: TableInfo) => {
    setMenuAnchorEl(event.currentTarget);
    setSelectedTableForMenu(table);
  };

  const handleMenuClose = () => {
    setMenuAnchorEl(null);
    setSelectedTableForMenu(null);
  };

  const handleViewSchema = (table: TableInfo) => {
    if (!table || !table.columns) {
      setError('Table schema is not available');
      return;
    }
    setSelectedTable(table);
    setShowSchema(true);
  };

  const handlePreviewData = (table: TableInfo) => {
    setSelectedTable(table);
    setShowPreview(true);
  };

  const handleViewFiles = (table: TableInfo) => {
    setSelectedTable(table);
    setShowFiles(true);
  };

  const handleExportCSV = async (table: TableInfo) => {
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
  };

  useEffect(() => {
    // Fetch file counts for each table
    const fetchFileCounts = async () => {
      try {
        const tables = await catalogService.listTables();
        const counts: Record<string, number> = {};
        
        // Fetch files for each table
        for (const table of tables) {
          try {
            const files = await catalogService.getTableFiles(table.name);
            counts[table.name] = files.length;
          } catch (err) {
            console.error(`Error fetching files for table ${table.name}:`, err);
            counts[table.name] = 0;
          }
        }
        
        setTableFileCounts(counts);
      } catch (err) {
        console.error('Error fetching file counts:', err);
      }
    };

    fetchFileCounts();
  }, []);

  const fetchSchema = async (tableName: string) => {
    try {
      // Check if schema is already in cache
      if (schemaCache[tableName]) {
        console.log('Using cached schema for table:', tableName);
        return schemaCache[tableName];
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
        console.log('Mapped schema before cache update:', mappedSchema);
        
        // Update cache
        setSchemaCache(prev => ({
          ...prev,
          [tableName]: mappedSchema
        }));
        console.log('Schema cache updated');
        return mappedSchema;
      } else {
        console.log('Invalid schema data format:', data);
        return [];
      }
    } catch (error) {
      console.error('Error in fetchSchema:', error);
      return [];
    }
  };

  const handleQueryExecute = (query: string) => {
    // Handle query execution
    console.log('Executing query:', query);
    // Add your query execution logic here
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* Left Panel - Tables List */}
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { 
            width: drawerWidth, 
            boxSizing: 'border-box',
            backgroundColor: '#f5f5f5'
          },
        }}
        anchor="left"
      >
        <Box sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>Tables</Typography>
            <Tooltip title="Refresh Tables">
              <IconButton onClick={fetchTables} size="small">
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>
          
          <TextField
            fullWidth
            size="small"
            placeholder="Search tables..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            InputProps={{
              startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
            }}
            sx={{ mb: 2 }}
          />

          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <CircularProgress size={24} />
            </Box>
          ) : (
            <List sx={{ 
              maxHeight: 'calc(100vh - 200px)', 
              overflow: 'auto',
              '& .MuiListItem-root': {
                borderRadius: 1,
                mb: 0.5,
                '&:hover': {
                  backgroundColor: 'rgba(0, 0, 0, 0.04)',
                },
              }
            }}>
              {filteredTables.map((table) => (
                <ListItem
                  button
                  key={table.name}
                  selected={selectedTables.some(t => t.name === table.name)}
                  onClick={() => handleTableSelect(table)}
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <ListItemText 
                    primary={
                      <Typography variant="body1" component="div">
                        <Box display="flex" alignItems="center">
                          {table.name}
                          {tableFileCounts[table.name] > 0 && (
                            <Tooltip title="Number of files">
                              <Chip
                                size="small"
                                icon={<FileIcon />}
                                label={tableFileCounts[table.name]}
                                sx={{ ml: 1 }}
                              />
                            </Tooltip>
                          )}
                        </Box>
                      </Typography>
                    }
                    secondary={
                      <Typography variant="body2" component="div" color="text.secondary">
                        {table.description || 'No description'}
                      </Typography>
                    }
                  />
                  <IconButton 
                    size="small" 
                    onClick={(e) => {
                      e.stopPropagation();
                      handleMenuOpen(e, table);
                    }}
                  >
                    <MoreVertIcon fontSize="small" />
                  </IconButton>
                </ListItem>
              ))}
            </List>
          )}
        </Box>
      </Drawer>

      {/* Main Content Area */}
      <Box sx={{ flexGrow: 1, ml: 0, p: 2, overflow: 'auto' }}>
        <Typography variant="h4" gutterBottom>
          Data Agent
        </Typography>

        {/* Selected Tables Section */}
        {selectedTables.length > 0 && (
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Selected Tables
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {selectedTables.map((table) => (
                <Chip
                  key={table.name}
                  label={table.name}
                  onDelete={() => handleTableRemove(table.name)}
                  color="primary"
                  variant="outlined"
                  sx={{ m: 0.5 }}
                />
              ))}
            </Box>
          </Paper>
        )}

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Paper sx={{ width: '100%' }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            indicatorColor="primary"
            textColor="primary"
          >
            <Tab icon={<UploadIcon />} label="Upload" />
            <Tab 
              icon={
                <Badge badgeContent={selectedTables.length} color="primary">
                  <TableIcon />
                </Badge>
              } 
              label="Explore" 
            />
            <Tab 
              icon={
                <Badge badgeContent={selectedTables.length} color="primary">
                  <QualityIcon />
                </Badge>
              } 
              label="Quality" 
            />
          </Tabs>

          <TabPanel value={activeTab} index={0}>
            <FileUpload
              onUploadSuccess={handleUploadSuccess}
              onError={handleError}
            />
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            <DataExplorer
              selectedTables={selectedTables}
              onTableSelect={handleTableSelect}
              onQueryExecute={handleQueryExecute}
            />
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            <QualityMetrics selectedTables={selectedTables} />
          </TabPanel>
        </Paper>
      </Box>

      {/* Schema Dialog */}
      <SchemaDialog
        open={showSchema}
        onClose={() => setShowSchema(false)}
        tableName={selectedTable?.name || ''}
        fetchSchema={fetchSchema}
      />

      {/* Preview Dialog */}
      {selectedTable && (
        <DataPreviewDialog
          open={showPreview}
          onClose={() => setShowPreview(false)}
          table={selectedTable}
        />
      )}

      {/* Files Dialog */}
      {selectedTable && (
        <TableFilesDialog
          open={showFiles}
          onClose={() => setShowFiles(false)}
          table={selectedTable}
        />
      )}

      {/* Table Actions Menu */}
      <Menu
        anchorEl={menuAnchorEl}
        open={Boolean(menuAnchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => {
          if (selectedTableForMenu) {
            setSelectedTable(selectedTableForMenu);
            setShowSchema(true);
          }
          handleMenuClose();
        }}>
          <SchemaIcon fontSize="small" sx={{ mr: 1 }} />
          View Schema
        </MenuItem>
        <MenuItem onClick={() => {
          if (selectedTableForMenu) {
            handleViewFiles(selectedTableForMenu);
          }
          handleMenuClose();
        }}>
          <FileIcon fontSize="small" sx={{ mr: 1 }} />
          View Files
        </MenuItem>
        <MenuItem onClick={() => {
          if (selectedTableForMenu) {
            handleExportCSV(selectedTableForMenu);
          }
          handleMenuClose();
        }}>
          <DownloadIcon fontSize="small" sx={{ mr: 1 }} />
          Export as CSV
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default CatalogDashboard; 