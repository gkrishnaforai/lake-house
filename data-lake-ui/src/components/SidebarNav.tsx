import React from 'react';
import { Box, List, ListItem, ListItemButton, ListItemIcon, ListItemText } from '@mui/material';
import ChatIcon from '@mui/icons-material/Chat';
import TableChartIcon from '@mui/icons-material/TableChart';
import BuildIcon from '@mui/icons-material/Build';
import HistoryIcon from '@mui/icons-material/History';
import { useLocation, useNavigate } from 'react-router-dom';
import logo from '../logo.svg'; // Adjust if your logo is in a different path

const navItems = [
  { label: 'Chat', icon: <ChatIcon />, path: '/chat' },
  { label: 'Catalog', icon: <TableChartIcon />, path: '/catalog' },
  { label: 'Infra', icon: <BuildIcon />, path: '/infra' },
  { label: 'Activity', icon: <HistoryIcon />, path: '/activity' },
];

const SidebarNav: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <Box sx={{ width: 220, bgcolor: 'white', borderRight: '1px solid #e0e0e0', display: 'flex', flexDirection: 'column', alignItems: 'center', pt: 2 }}>
      <Box sx={{ mb: 4 }}>
        <img src={logo} alt="Logo" style={{ width: 48, height: 48 }} />
      </Box>
      <List sx={{ width: '100%' }}>
        {navItems.map(item => (
          <ListItem key={item.label} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => navigate(item.path)}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );
};

export default SidebarNav; 