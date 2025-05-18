import React from 'react';
import { Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import CatalogDashboard from './components/CatalogDashboard';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', py: 2 }}>
        <CatalogDashboard />
      </Box>
    </ThemeProvider>
  );
};

export default App; 