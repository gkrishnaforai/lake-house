import React from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
  Box,
  Typography,
  TextField,
  InputAdornment,
  IconButton,
  Badge,
  Tooltip,
  Button
} from '@mui/material';
import {
  Search as SearchIcon,
  Refresh as RefreshIcon,
  TableChart as TableIcon,
  Dashboard as DashboardIcon,
  Assessment as QualityIcon,
  Explore as ExploreIcon,
  Add as AddIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { TableInfo } from '../../types/api';

interface SidebarProps {
  open: boolean;
  onClose: () => void;
  variant: 'permanent' | 'temporary';
  width: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  open,
  onClose,
  variant,
  width
}) => {
  const navigate = useNavigate();
  const [search, setSearch] = React.useState('');
  const [tables, setTables] = React.useState<TableInfo[]>([]);
  const [selectedTables, setSelectedTables] = React.useState<TableInfo[]>([]);
  const [loading, setLoading] = React.useState(false);

  const handleTableSelect = (table: TableInfo) => {
    setSelectedTables(prev => {
      const exists = prev.find(t => t.name === table.name);
      if (exists) {
        return prev.filter(t => t.name !== table.name);
      }
      return [...prev, table];
    });
  };

  const filteredTables = tables.filter(table =>
    table.name.toLowerCase().includes(search.toLowerCase()) ||
    table.description?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <Drawer
      variant={variant}
      open={open}
      onClose={onClose}
      sx={{
        width: width,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: width,
          boxSizing: 'border-box',
          backgroundColor: 'background.default',
          borderRight: '1px solid',
          borderColor: 'divider'
        },
      }}
    >
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Data Lakehouse
        </Typography>
        
        <Button
          fullWidth
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => navigate('/explore')}
          sx={{ mb: 2 }}
        >
          New Analysis
        </Button>

        <TextField
          fullWidth
          size="small"
          placeholder="Search tables..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon fontSize="small" />
              </InputAdornment>
            ),
            endAdornment: (
              <InputAdornment position="end">
                <Tooltip title="Refresh Tables">
                  <IconButton size="small" onClick={() => {}}>
                    <RefreshIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </InputAdornment>
            ),
          }}
          sx={{ mb: 2 }}
        />

        <List sx={{ 
          maxHeight: 'calc(100vh - 300px)', 
          overflow: 'auto',
          '& .MuiListItem-root': {
            borderRadius: 1,
            mb: 0.5,
            '&:hover': {
              backgroundColor: 'action.hover',
            },
          }
        }}>
          {filteredTables.map((table) => (
            <ListItem
              key={table.name}
              disablePadding
              secondaryAction={
                <Badge
                  badgeContent={selectedTables.find(t => t.name === table.name) ? 1 : 0}
                  color="primary"
                >
                  <TableIcon fontSize="small" />
                </Badge>
              }
            >
              <ListItemButton
                selected={selectedTables.some(t => t.name === table.name)}
                onClick={() => handleTableSelect(table)}
              >
                <ListItemText
                  primary={table.name}
                  secondary={table.description || 'No description'}
                  primaryTypographyProps={{
                    variant: 'body2',
                    fontWeight: selectedTables.some(t => t.name === table.name) ? 'bold' : 'normal'
                  }}
                  secondaryTypographyProps={{
                    variant: 'caption',
                    color: 'text.secondary'
                  }}
                />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>

      <Divider />

      <List>
        <ListItem disablePadding>
          <ListItemButton onClick={() => navigate('/')}>
            <ListItemIcon>
              <DashboardIcon />
            </ListItemIcon>
            <ListItemText primary="Dashboard" />
          </ListItemButton>
        </ListItem>
        <ListItem disablePadding>
          <ListItemButton onClick={() => navigate('/explore')}>
            <ListItemIcon>
              <ExploreIcon />
            </ListItemIcon>
            <ListItemText primary="Data Explorer" />
          </ListItemButton>
        </ListItem>
        <ListItem disablePadding>
          <ListItemButton onClick={() => navigate('/quality')}>
            <ListItemIcon>
              <QualityIcon />
            </ListItemIcon>
            <ListItemText primary="Data Quality" />
          </ListItemButton>
        </ListItem>
      </List>
    </Drawer>
  );
}; 