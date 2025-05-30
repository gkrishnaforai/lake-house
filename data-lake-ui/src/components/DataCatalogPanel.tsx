import React from 'react';
import { Paper, Typography } from '@mui/material';

const DataCatalogPanel: React.FC = () => (
  <Paper sx={{ maxWidth: 700, margin: '0 auto', p: 3, minHeight: 500 }}>
    <Typography variant="h6">Data Catalog</Typography>
    <Typography variant="body1" color="text.secondary">Catalog explorer coming soon...</Typography>
  </Paper>
);

export default DataCatalogPanel; 