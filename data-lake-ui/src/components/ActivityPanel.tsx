import React from 'react';
import { Paper, Typography } from '@mui/material';

const ActivityPanel: React.FC = () => (
  <Paper sx={{ maxWidth: 700, margin: '0 auto', p: 3, minHeight: 500 }}>
    <Typography variant="h6">Activity</Typography>
    <Typography variant="body1" color="text.secondary">Recent activity and logs coming soon...</Typography>
  </Paper>
);

export default ActivityPanel; 