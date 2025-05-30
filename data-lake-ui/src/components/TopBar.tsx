import React from 'react';
import { AppBar, Toolbar, Typography, Box, IconButton } from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';

const TopBar: React.FC = () => {
  return (
    <AppBar position="static" color="inherit" elevation={0} sx={{ borderBottom: '1px solid #e0e0e0' }}>
      <Toolbar>
        <Typography variant="h6" color="primary" sx={{ flex: 1, fontWeight: 700 }}>
          Data Engineer Agent
        </Typography>
        <Box>
          {/* Quick actions can go here */}
        </Box>
        <IconButton color="inherit">
          <AccountCircleIcon />
        </IconButton>
        <IconButton color="inherit">
          <SettingsIcon />
        </IconButton>
      </Toolbar>
    </AppBar>
  );
};

export default TopBar; 