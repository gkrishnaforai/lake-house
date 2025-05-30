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
  Button,
  Stack,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  Chip,
  Tooltip,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  CheckCircle as SuccessIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  History as HistoryIcon,
  FileCopy as FileIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import { CatalogService, getUserId } from '../services/catalogService';
import { TableInfo, QualityMetrics as QualityMetricsType, FileInfo } from '../types/api';
import { QualityCheckConfig, QualityCheckConfig as QualityCheckConfigType } from './QualityCheckConfig';

interface QualityMetricsProps {
  selectedTables: TableInfo[];
}

interface QualityMetric {
  name: string;
  score: number;
  status: 'success' | 'warning' | 'error';
  details?: {
    total_cells?: number;
    null_cells?: number;
    total_rows?: number;
    duplicate_rows?: number;
    total_columns?: number;
    consistent_columns?: number;
    threshold?: number;
    total?: number;
    valid?: number;
    invalid?: number;
    missing?: number;
  };
}

interface FileMetric {
  fileName: string;
  metrics: QualityMetric[];
  uploadDate: string;
  size: number;
}

interface TableMetrics {
  metrics: Record<string, {
    score: number;
    status: 'success' | 'warning' | 'error';
    details?: {
      total_cells?: number;
      null_cells?: number;
      total_rows?: number;
      duplicate_rows?: number;
      total_columns?: number;
      consistent_columns?: number;
      threshold?: number;
      total?: number;
      valid?: number;
      invalid?: number;
      missing?: number;
    };
  }>;
  metadata?: {
    table_name: string;
    checked_at: string;
  };
}

const QualityMetrics: React.FC<QualityMetricsProps> = ({ selectedTables }) => {
  const [metrics, setMetrics] = useState<Record<string, TableMetrics>>({});
  const [fileMetrics, setFileMetrics] = useState<Record<string, FileMetric[]>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runQualityChecks, setRunQualityChecks] = useState(false);
  const [qualityConfig, setQualityConfig] = useState<QualityCheckConfigType>({
    enabled_metrics: ["completeness", "uniqueness", "consistency"],
    thresholds: {
      completeness: 0.95,
      uniqueness: 0.90,
      consistency: 0.85
    }
  });
  const [selectedTableIndex, setSelectedTableIndex] = useState(0);
  const [expandedAccordion, setExpandedAccordion] = useState<string | false>(false);
  const catalogService = new CatalogService();

  const fetchMetrics = async (tableName: string) => {
    setLoading(true);
    setError(null);

    try {
      const userId = getUserId();
      const response = await catalogService.getTableQuality(tableName);
      
      // Transform the response to match our TableMetrics interface
      const tableMetrics: TableMetrics = {
        metrics: response.metrics || {},
        metadata: {
          table_name: tableName,
          checked_at: new Date().toISOString()
        }
      };
      
      setMetrics(prev => ({
        ...prev,
        [tableName]: tableMetrics
      }));

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch quality metrics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedTables.length > 0) {
      const table = selectedTables[selectedTableIndex];
      fetchMetrics(table.name);
    }
  }, [selectedTables, selectedTableIndex]);

  const handleRunQualityChecks = async () => {
    if (selectedTables.length === 0) return;

    setLoading(true);
    setError(null);

    try {
      const userId = getUserId();
      const table = selectedTables[selectedTableIndex];
      await catalogService.configureQualityChecks(table.name, qualityConfig);
      await catalogService.runQualityChecks(table.name);
      await fetchMetrics(table.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run quality checks');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'success.main';
      case 'warning':
        return 'warning.main';
      case 'error':
        return 'error.main';
      default:
        return 'text.secondary';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <SuccessIcon />;
      case 'warning':
        return <WarningIcon />;
      case 'error':
        return <ErrorIcon />;
      default:
        return null;
    }
  };

  const getTrendIcon = (trend?: string) => {
    switch (trend) {
      case 'up':
        return <TrendingUpIcon color="success" />;
      case 'down':
        return <TrendingDownIcon color="error" />;
      default:
        return null;
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const renderMetricDetails = (details: any) => {
    if (!details) return null;
    
    return (
      <Box sx={{ mt: 2 }}>
        {details.total_cells !== undefined && (
          <>
            <Typography variant="body2" color="text.secondary">
              Total Cells: {details.total_cells}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Null Cells: {details.null_cells}
            </Typography>
          </>
        )}
        {details.total_rows !== undefined && (
          <>
            <Typography variant="body2" color="text.secondary">
              Total Rows: {details.total_rows}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Duplicate Rows: {details.duplicate_rows}
            </Typography>
          </>
        )}
        {details.total_columns !== undefined && (
          <>
            <Typography variant="body2" color="text.secondary">
              Total Columns: {details.total_columns}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Consistent Columns: {details.consistent_columns}
            </Typography>
          </>
        )}
        {details.threshold !== undefined && (
          <Typography variant="body2" color="text.secondary">
            Threshold: {(details.threshold * 100).toFixed(0)}%
          </Typography>
        )}
        {details.total !== undefined && (
          <>
            <Typography variant="body2" color="text.secondary">
              Total Records: {details.total_rows}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Valid: {details.valid}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Invalid: {details.invalid}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Missing: {details.missing}
            </Typography>
          </>
        )}
      </Box>
    );
  };

  if (selectedTables.length === 0) {
    return (
      <Box>
        <Typography color="text.secondary">
          Select tables to view their quality metrics.
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

  const currentTable = selectedTables[selectedTableIndex];
  const currentMetrics = metrics[currentTable.name];
  const currentFileMetrics = fileMetrics[currentTable.name] || [];

  return (
    <Box>
      <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 3 }}>
        <Typography variant="h6">
          Quality Metrics
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={handleRunQualityChecks}
          disabled={loading}
        >
          Run Quality Checks
        </Button>
      </Stack>

      {selectedTables.length > 1 && (
        <Tabs
          value={selectedTableIndex}
          onChange={(_, newValue) => setSelectedTableIndex(newValue)}
          sx={{ mb: 3 }}
        >
          {selectedTables.map((table, index) => (
            <Tab key={table.name} label={table.name} />
          ))}
        </Tabs>
      )}

      <QualityCheckConfig
        config={qualityConfig}
        onChange={(config) => setQualityConfig(config)}
        enabled={runQualityChecks}
        onEnabledChange={setRunQualityChecks}
      />

      {currentMetrics && (
        <>
          {/* Overall Metrics */}
          <Typography variant="h6" sx={{ mt: 4, mb: 2 }}>
            Overall Table Metrics
          </Typography>
          <Grid container spacing={3}>
            {Object.entries(currentMetrics.metrics).map(([metricName, metricData]) => (
              <Grid item xs={12} md={4} key={metricName}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Box sx={{ 
                        color: getStatusColor(metricData.status),
                        mr: 1
                      }}>
                        {getStatusIcon(metricData.status)}
                      </Box>
                      <Typography variant="h6" component="div">
                        {metricName.charAt(0).toUpperCase() + metricName.slice(1)}
                      </Typography>
                    </Box>
                    <Typography variant="h4" component="div" sx={{ mb: 1 }}>
                      {(metricData.score * 100).toFixed(1)}%
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={metricData.score * 100}
                      color={metricData.status === 'success' ? 'success' : metricData.status === 'warning' ? 'warning' : 'error'}
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                    {renderMetricDetails(metricData.details)}
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                      Last updated: {currentMetrics.metadata?.checked_at ? new Date(currentMetrics.metadata.checked_at).toLocaleString() : 'N/A'}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>

          {/* Configuration Information */}
          <Typography variant="h6" sx={{ mt: 4, mb: 2 }}>
            Configuration
          </Typography>
          <Paper sx={{ p: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Enabled Metrics
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {Object.keys(currentMetrics.metrics).map(metric => (
                    <Chip
                      key={metric}
                      label={metric.charAt(0).toUpperCase() + metric.slice(1)}
                      color="primary"
                      variant="outlined"
                    />
                  ))}
                </Box>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Thresholds
                </Typography>
                {Object.entries(qualityConfig.thresholds).map(([metric, threshold]) => (
                  <Box key={metric} sx={{ mb: 1 }}>
                    <Typography variant="body2">
                      {metric.charAt(0).toUpperCase() + metric.slice(1)}: {(threshold * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                ))}
              </Grid>
            </Grid>
          </Paper>
        </>
      )}

      {/* File-level Metrics */}
      <Typography variant="h6" sx={{ mt: 4, mb: 2 }}>
        File-level Metrics
      </Typography>
      {currentFileMetrics.length > 0 ? (
        currentFileMetrics.map((fileMetric, index) => (
          <Accordion
            key={fileMetric.fileName}
            expanded={expandedAccordion === fileMetric.fileName}
            onChange={(_, isExpanded) => setExpandedAccordion(isExpanded ? fileMetric.fileName : false)}
            sx={{ mb: 2 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                <FileIcon sx={{ mr: 1 }} />
                <Typography sx={{ flex: 1 }}>
                  {fileMetric.fileName}
                </Typography>
                <Chip
                  label={formatFileSize(fileMetric.size)}
                  size="small"
                  sx={{ mr: 1 }}
                />
                <Typography variant="caption" color="text.secondary">
                  {new Date(fileMetric.uploadDate).toLocaleString()}
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {fileMetric.metrics.map((metric: QualityMetric) => (
                  <Grid item xs={12} md={4} key={metric.name}>
                    <Card variant="outlined">
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <Box sx={{ 
                            color: getStatusColor(metric.status),
                            mr: 1
                          }}>
                            {getStatusIcon(metric.status)}
                          </Box>
                          <Typography variant="subtitle1">
                            {metric.name}
                          </Typography>
                        </Box>
                        <Typography variant="h6" component="div">
                          {metric.score.toFixed(1)}%
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={metric.score}
                          color={metric.status === 'success' ? 'success' : metric.status === 'warning' ? 'warning' : 'error'}
                          sx={{ height: 6, borderRadius: 3 }}
                        />
                        {metric.details && (
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="caption" color="text.secondary">
                              Valid: {metric.details.valid} / {metric.details.total}
                            </Typography>
                          </Box>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        ))
      ) : (
        <Typography color="text.secondary">
          No file-level metrics available for this table.
        </Typography>
      )}
    </Box>
  );
};

export default QualityMetrics; 