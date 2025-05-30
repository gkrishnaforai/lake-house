import React from 'react';
import { Box } from '@mui/material';
import SidebarNav from './SidebarNav';
import TopBar from './TopBar';

const AppLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <Box sx={{ display: 'flex', height: '100vh', background: '#f7f9fb' }}>
      <SidebarNav />
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <TopBar />
        <Box sx={{ flex: 1, overflow: 'auto', p: 3 }}>{children}</Box>
      </Box>
    </Box>
  );
};

export default AppLayout; 