import React from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Tooltip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Button,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  TableChart as TableIcon,
  Assessment as QualityIcon,
  FilterList as FilterIcon,
  Download as DownloadIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon
} from '@mui/icons-material';

interface QualityMetric {
  name: string;
  value: number;
  trend: 'up' | 'down' | 'stable';
  status: 'success' | 'warning' | 'error';
}

interface QualityIssue {
  id: string;
  table: string;
  type: string;
  severity: 'high' | 'medium' | 'low';
  status: 'open' | 'in_progress' | 'resolved';
  lastUpdated: string;
}

const QualityMetricCard: React.FC<{ metric: QualityMetric }> = ({ metric }) => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Box sx={{ 
          backgroundColor: `${metric.status === 'success' ? '#2e7d32' : metric.status === 'warning' ? '#ed6c02' : '#d32f2f'}15`, 
          borderRadius: 1, 
          p: 1,
          mr: 2
        }}>
          <QualityIcon sx={{ 
            color: metric.status === 'success' ? 'success.main' : 
                   metric.status === 'warning' ? 'warning.main' : 'error.main' 
          }} />
        </Box>
        <Typography variant="h6" component="div">
          {metric.name}
        </Typography>
      </Box>
      <Typography variant="h4" component="div" sx={{ mb: 1 }}>
        {metric.value}%
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        {metric.trend === 'up' ? (
          <TrendingUpIcon color="success" fontSize="small" />
        ) : metric.trend === 'down' ? (
          <TrendingDownIcon color="error" fontSize="small" />
        ) : null}
        <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
          {metric.trend === 'up' ? 'Improving' : 
           metric.trend === 'down' ? 'Declining' : 'Stable'}
        </Typography>
      </Box>
    </CardContent>
  </Card>
);

export const DataQuality: React.FC = () => {
  const [loading, setLoading] = React.useState(false);
  const [page, setPage] = React.useState(0);
  const [rowsPerPage, setRowsPerPage] = React.useState(10);

  const handleRefresh = () => {
    setLoading(true);
    // TODO: Implement refresh logic
    setTimeout(() => setLoading(false), 1000);
  };

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const qualityMetrics: QualityMetric[] = [
    {
      name: 'Overall Quality Score',
      value: 92,
      trend: 'up',
      status: 'success'
    },
    {
      name: 'Completeness',
      value: 95,
      trend: 'up',
      status: 'success'
    },
    {
      name: 'Accuracy',
      value: 88,
      trend: 'down',
      status: 'warning'
    },
    {
      name: 'Consistency',
      value: 90,
      trend: 'stable',
      status: 'success'
    }
  ];

  const qualityIssues: QualityIssue[] = [
    {
      id: '1',
      table: 'Customer Data',
      type: 'Missing Values',
      severity: 'high',
      status: 'open',
      lastUpdated: '2 hours ago'
    },
    {
      id: '2',
      table: 'Sales Transactions',
      type: 'Data Type Mismatch',
      severity: 'medium',
      status: 'in_progress',
      lastUpdated: '5 hours ago'
    },
    {
      id: '3',
      table: 'Product Catalog',
      type: 'Duplicate Records',
      severity: 'low',
      status: 'resolved',
      lastUpdated: '1 day ago'
    }
  ];

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Data Quality</Typography>
        <Box>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            sx={{ mr: 1 }}
          >
            Export Report
          </Button>
          <Tooltip title="Refresh">
            <IconButton onClick={handleRefresh} disabled={loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {loading && <LinearProgress sx={{ mb: 3 }} />}

      <Grid container spacing={3}>
        {qualityMetrics.map((metric, index) => (
          <Grid item xs={12} md={3} key={index}>
            <QualityMetricCard metric={metric} />
          </Grid>
        ))}

        <Grid item xs={12}>
          <Card>
            <CardHeader
              title="Quality Issues"
              action={
                <Tooltip title="Filter">
                  <IconButton>
                    <FilterIcon />
                  </IconButton>
                </Tooltip>
              }
            />
            <CardContent>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Table</TableCell>
                      <TableCell>Issue Type</TableCell>
                      <TableCell>Severity</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Last Updated</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {qualityIssues
                      .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                      .map((issue) => (
                        <TableRow key={issue.id}>
                          <TableCell>{issue.table}</TableCell>
                          <TableCell>{issue.type}</TableCell>
                          <TableCell>
                            <Chip
                              label={issue.severity}
                              color={
                                issue.severity === 'high' ? 'error' :
                                issue.severity === 'medium' ? 'warning' : 'default'
                              }
                              size="small"
                            />
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={issue.status}
                              color={
                                issue.status === 'resolved' ? 'success' :
                                issue.status === 'in_progress' ? 'warning' : 'default'
                              }
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{issue.lastUpdated}</TableCell>
                          <TableCell>
                            <Button size="small">View Details</Button>
                          </TableCell>
                        </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              <TablePagination
                rowsPerPageOptions={[5, 10, 25]}
                component="div"
                count={qualityIssues.length}
                rowsPerPage={rowsPerPage}
                page={page}
                onPageChange={handleChangePage}
                onRowsPerPageChange={handleChangeRowsPerPage}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}; 