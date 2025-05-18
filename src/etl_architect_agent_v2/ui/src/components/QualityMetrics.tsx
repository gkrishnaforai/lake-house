import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  LinearProgress,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  CheckCircle as SuccessIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { CatalogService } from '../services/catalogService';
import { TableInfo } from '../types/api';

interface QualityMetricsProps {
  selectedTables: TableInfo[];
}

interface QualityMetric {
  name: string;
  score: number;
  description: string;
  details?: {
    total: number;
    valid: number;
    invalid: number;
    missing: number;
  };
}

const QualityMetrics: React.FC<QualityMetricsProps> = ({ selectedTables }) => {
  const [metrics, setMetrics] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const catalogService = new CatalogService();

  useEffect(() => {
    const fetchMetrics = async () => {
      if (selectedTables.length === 0) {
        setMetrics({});
        return;
      }

      setLoading(true);
      setError(null);
      const newMetrics: Record<string, any> = {};

      try {
        for (const table of selectedTables) {
          const data = await catalogService.getQualityMetrics(table.name);
          newMetrics[table.name] = data;
        }
        setMetrics(newMetrics);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch quality metrics');
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, [selectedTables]);

  if (selectedTables.length === 0) {
    return (
      <Box>
        <Typography color="text.secondary">
          Select tables from the left panel to view their quality metrics.
        </Typography>
      </Box>
    );
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      {selectedTables.map((table) => (
        <Box key={table.name} sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Quality Metrics for {table.name}
          </Typography>
          {metrics[table.name] ? (
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Completeness
                  </Typography>
                  <Typography variant="h4">
                    {metrics[table.name].completeness?.toFixed(2)}%
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} md={4}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Accuracy
                  </Typography>
                  <Typography variant="h4">
                    {metrics[table.name].accuracy?.toFixed(2)}%
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} md={4}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Consistency
                  </Typography>
                  <Typography variant="h4">
                    {metrics[table.name].consistency?.toFixed(2)}%
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
          ) : (
            <Typography color="text.secondary">
              No quality metrics available for this table.
            </Typography>
          )}
        </Box>
      ))}
    </Box>
  );
};

export default QualityMetrics; 