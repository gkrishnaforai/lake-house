import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  CircularProgress,
  Alert,
  Card,
  CardContent,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { DataQualityDashboardProps } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const DataQualityDashboard: React.FC<DataQualityDashboardProps> = ({ selectedTable, selectedFile }) => {
  const { data: qualityMetrics, isLoading, error } = useQuery({
    queryKey: ['quality', selectedTable || selectedFile],
    queryFn: async () => {
      const target = selectedTable || selectedFile;
      if (!target) return null;
      const response = await axios.get(`${API_BASE_URL}/api/catalog/quality/${target}`);
      return response.data;
    },
    enabled: !!selectedTable || !!selectedFile,
  });

  if (!selectedTable && !selectedFile) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="info">Select a table or file to view quality metrics</Alert>
      </Box>
    );
  }

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error instanceof Error) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="error">Error loading quality metrics: {error.message}</Alert>
      </Box>
    );
  }

  if (!qualityMetrics) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="warning">No quality metrics available</Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Data Quality Metrics
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Completeness
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {qualityMetrics.completeness}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Accuracy
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {qualityMetrics.accuracy}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Consistency
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {qualityMetrics.consistency}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Timeliness
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {qualityMetrics.timeliness}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DataQualityDashboard; 