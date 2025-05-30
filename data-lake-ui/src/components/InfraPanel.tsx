import React from 'react';
import { Paper, Typography } from '@mui/material';

const InfraPanel: React.FC = () => (
  <Paper sx={{ maxWidth: 700, margin: '0 auto', p: 3, minHeight: 500 }}>
    <Typography variant="h6">Infrastructure</Typography>
    <Typography variant="body1" color="text.secondary">Infrastructure management coming soon...</Typography>
  </Paper>
);

export default InfraPanel; 