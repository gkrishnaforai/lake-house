import React from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  IconButton,
  Tooltip,
  Tabs,
  Tab,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Menu,
  MenuItem
} from '@mui/material';
import {
  Search as SearchIcon,
  Add as AddIcon,
  FilterList as FilterIcon,
  Sort as SortIcon,
  TableChart as TableIcon,
  Assessment as QualityIcon,
  Code as CodeIcon,
  Download as DownloadIcon,
  Share as ShareIcon,
  MoreVert as MoreIcon
} from '@mui/icons-material';
import { TableInfo } from '../types/api';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

interface DataExplorerProps {
  selectedTables: TableInfo[];
  onTableSelect: (table: TableInfo) => void;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div
    role="tabpanel"
    hidden={value !== index}
    id={`tabpanel-${index}`}
    aria-labelledby={`tab-${index}`}
  >
    {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
  </div>
);

export const DataExplorer: React.FC<DataExplorerProps> = ({ selectedTables, onTableSelect }) => {
  const [search, setSearch] = React.useState('');
  const [tabValue, setTabValue] = React.useState(0);
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const [selectedTable, setSelectedTable] = React.useState<TableInfo | null>(null);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleTableClick = (table: TableInfo) => {
    setSelectedTable(table);
    onTableSelect(table);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Data Explorer</Typography>
        <Box>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            sx={{ mr: 1 }}
            onClick={() => {/* TODO: Implement new analysis */}}
          >
            New Analysis
          </Button>
          <IconButton onClick={handleMenuClick}>
            <MoreIcon />
          </IconButton>
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
          >
            <MenuItem onClick={handleMenuClose}>
              <ListItemIcon>
                <DownloadIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Export Data</ListItemText>
            </MenuItem>
            <MenuItem onClick={handleMenuClose}>
              <ListItemIcon>
                <ShareIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Share Analysis</ListItemText>
            </MenuItem>
          </Menu>
        </Box>
      </Box>

      <Paper sx={{ mb: 3 }}>
        <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
          <TextField
            fullWidth
            placeholder="Search tables, columns, or data..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            InputProps={{
              startAdornment: (
                <SearchIcon sx={{ color: 'text.secondary', mr: 1 }} />
              ),
            }}
          />
          <Tooltip title="Filter">
            <IconButton>
              <FilterIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Sort">
            <IconButton>
              <SortIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Paper>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
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
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={tabValue} onChange={handleTabChange}>
                <Tab label="Data Preview" />
                <Tab label="Schema" />
                <Tab label="Quality Metrics" />
                <Tab label="SQL Editor" />
              </Tabs>
            </Box>

            <TabPanel value={tabValue} index={0}>
              {selectedTable ? (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    {selectedTable.name} Preview
                  </Typography>
                  {/* TODO: Add data preview table */}
                </Box>
              ) : (
                <Typography variant="body1" color="text.secondary" gutterBottom>
                  Select a table to preview its data
                </Typography>
              )}
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              {selectedTable ? (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    {selectedTable.name} Schema
                  </Typography>
                  <List>
                    {selectedTable.columns?.map((column) => (
                      <ListItem key={column.name}>
                        <ListItemText
                          primary={column.name}
                          secondary={`${column.type} - ${column.description || 'No description'}`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              ) : (
                <Typography variant="body1" color="text.secondary" gutterBottom>
                  Select a table to view its schema
                </Typography>
              )}
            </TabPanel>

            <TabPanel value={tabValue} index={2}>
              {selectedTable ? (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    {selectedTable.name} Quality Metrics
                  </Typography>
                  {/* TODO: Add quality metrics visualization */}
                </Box>
              ) : (
                <Typography variant="body1" color="text.secondary" gutterBottom>
                  Select a table to view its quality metrics
                </Typography>
              )}
            </TabPanel>

            <TabPanel value={tabValue} index={3}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  multiline
                  rows={4}
                  placeholder="Write your SQL query here..."
                  fullWidth
                />
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1 }}>
                  <Button variant="outlined" startIcon={<CodeIcon />}>
                    Format SQL
                  </Button>
                  <Button variant="contained" startIcon={<TableIcon />}>
                    Run Query
                  </Button>
                </Box>
              </Box>
            </TabPanel>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}; 