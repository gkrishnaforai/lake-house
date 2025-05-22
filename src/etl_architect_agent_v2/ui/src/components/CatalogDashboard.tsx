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
  TableRow
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
  InsertDriveFile as FileIcon
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
  table: TableInfo | null;
  userId: string;
}> = ({ open, onClose, table, userId }) => {
  const catalogService = new CatalogService();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [schema, setSchema] = useState<any>(null);

  useEffect(() => {
    const fetchSchema = async () => {
      if (!table) return;
      setLoading(true);
      setError(null);
      try {
        const schemaData = await catalogService.getTableSchema(table.name);
        setSchema(schemaData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch schema');
      } finally {
        setLoading(false);
      }
    };

    if (open && table) {
      fetchSchema();
    }
  }, [open, table]);

  const formatDate = (dateString: string | undefined): string => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleDateString();
    } catch (e) {
      return 'Invalid Date';
    }
  };

  if (!table) return null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" alignItems="center">
          <SchemaIcon sx={{ mr: 1 }} />
          Schema for {table.name}
        </Box>
      </DialogTitle>
      <DialogContent dividers>
        <Grid container spacing={3}>
          {/* Basic Info */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>Description</Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              {table.description || 'No description available'}
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <Chip 
                label={`Created: ${formatDate(table.created_at)}`} 
                size="small" 
                variant="outlined" 
              />
              <Chip 
                label={`Updated: ${formatDate(table.updated_at)}`} 
                size="small" 
                variant="outlined" 
              />
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Divider />
          </Grid>

          {/* Schema */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>Schema</Typography>
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                <CircularProgress size={24} />
              </Box>
            ) : error ? (
              <Alert severity="error">{error}</Alert>
            ) : schema?.schema ? (
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Column Name</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Description</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {schema.schema.map((col: any) => (
                      <TableRow key={col.name}>
                        <TableCell>{col.name}</TableCell>
                        <TableCell>{col.type}</TableCell>
                        <TableCell>{col.comment || '-'}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No schema information available
              </Typography>
            )}
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
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
    setLoading(true);
    try {
      const data = await catalogService.listTables(userId);
      setTables(data);
      setFilteredTables(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch tables');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTables();
  }, [userId]);

  useEffect(() => {
    if (!search) {
      setFilteredTables(tables);
    } else {
      setFilteredTables(
        tables.filter((t) => 
          t.name.toLowerCase().includes(search.toLowerCase()) ||
          t.description?.toLowerCase().includes(search.toLowerCase())
        )
      );
    }
  }, [search, tables]);

  const handleTableSelect = (table: TableInfo) => {
    setSelectedTables(prev => {
      const exists = prev.find(t => t.name === table.name);
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

  useEffect(() => {
    // Fetch file counts for each table
    const fetchFileCounts = async () => {
      try {
        const files = await catalogService.listFiles();
        const counts: Record<string, number> = {};
        files.forEach(file => {
          const tableName = file.table_name;
          if (tableName) {
            counts[tableName] = (counts[tableName] || 0) + 1;
          }
        });
        setTableFileCounts(counts);
      } catch (err) {
        console.error('Error fetching file counts:', err);
      }
    };

    fetchFileCounts();
  }, []);

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
                    }
                    secondary={table.description || 'No description'}
                    primaryTypographyProps={{
                      variant: 'body1',
                      fontWeight: selectedTables.some(t => t.name === table.name) ? 'bold' : 'normal'
                    }}
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
        table={selectedTable}
        userId={userId}
      />

      {/* Preview Dialog */}
      {selectedTable && (
        <DataPreviewDialog
          open={showPreview}
          onClose={() => setShowPreview(false)}
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
            setSelectedTable(selectedTableForMenu);
            setShowPreview(true);
          }
          handleMenuClose();
        }}>
          <PreviewIcon fontSize="small" sx={{ mr: 1 }} />
          Preview Data
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default CatalogDashboard; 