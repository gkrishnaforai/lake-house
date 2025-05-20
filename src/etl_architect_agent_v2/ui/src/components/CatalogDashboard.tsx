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
  Badge
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
  Refresh as RefreshIcon
} from '@mui/icons-material';
import FileUpload from './FileUpload';
import DataExplorer from './DataExplorer';
import SchemaViewer from './SchemaViewer';
import QualityMetrics from './QualityMetrics';
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

const CatalogDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedTables, setSelectedTables] = useState<TableInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tables, setTables] = useState<TableInfo[]>([]);
  const [filteredTables, setFilteredTables] = useState<TableInfo[]>([]);
  const [search, setSearch] = useState('');
  const [userId, setUserId] = useState('test_user');
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
                    primary={table.name}
                    secondary={table.description || 'No description'}
                    primaryTypographyProps={{
                      variant: 'body1',
                      fontWeight: selectedTables.some(t => t.name === table.name) ? 'bold' : 'normal'
                    }}
                  />
                  <IconButton size="small" sx={{ ml: 1 }}>
                    <InfoIcon fontSize="small" />
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

        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Upload Data
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Upload your data files to start exploring and analyzing.
                </Typography>
              </CardContent>
              <CardActions>
                <Button
                  startIcon={<UploadIcon />}
                  onClick={() => setActiveTab(0)}
                  variant="contained"
                >
                  Upload Files
                </Button>
              </CardActions>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Explore Data
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Browse and query your data using SQL.
                </Typography>
              </CardContent>
              <CardActions>
                <Button
                  startIcon={<TableIcon />}
                  onClick={() => setActiveTab(1)}
                  variant="contained"
                  disabled={selectedTables.length === 0}
                >
                  View Data
                </Button>
              </CardActions>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Data Quality
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Monitor and analyze data quality metrics.
                </Typography>
              </CardContent>
              <CardActions>
                <Button
                  startIcon={<QualityIcon />}
                  onClick={() => setActiveTab(2)}
                  variant="contained"
                  disabled={selectedTables.length === 0}
                >
                  Check Quality
                </Button>
              </CardActions>
            </Card>
          </Grid>
        </Grid>

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
                  <SchemaIcon />
                </Badge>
              } 
              label="Schema" 
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
            <SchemaViewer selectedTables={selectedTables} />
          </TabPanel>

          <TabPanel value={activeTab} index={3}>
            <QualityMetrics selectedTables={selectedTables} />
          </TabPanel>
        </Paper>
      </Box>
    </Box>
  );
};

export default CatalogDashboard; 