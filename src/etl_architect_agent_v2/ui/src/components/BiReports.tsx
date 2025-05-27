import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Menu,
  MenuItem,
  Button,
  TextField,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { MoreVert as MoreVertIcon, Refresh as RefreshIcon } from '@mui/icons-material';
import { CatalogService } from '../services/catalogService';

// Define chart types
type ChartType = 'bar' | 'line' | 'pie';

// Define chart configuration
interface ChartConfig {
  id: string;
  title: string;
  type: ChartType;
  query: string;
  xField: string;
  yField: string;
  color?: string;
}

// Sample chart configurations
const defaultCharts: ChartConfig[] = [
  {
    id: 'funding-by-company',
    title: 'Funding by Company',
    type: 'bar',
    query: 'SELECT organization_name, last_funding_amount_in_usd FROM user_test_user.sample11111 WHERE last_funding_amount_in_usd IS NOT NULL ORDER BY last_funding_amount_in_usd DESC LIMIT 10',
    xField: 'organization_name',
    yField: 'last_funding_amount_in_usd',
    color: '#1976d2',
  },
  {
    id: 'employee-distribution',
    title: 'Employee Distribution',
    type: 'pie',
    query: 'SELECT empl_count, COUNT(*) as count FROM user_test_user.sample11111 GROUP BY empl_count',
    xField: 'empl_count',
    yField: 'count',
  },
  {
    id: 'funding-trend',
    title: 'Funding Trend',
    type: 'line',
    query: 'SELECT last_funding_date, SUM(last_funding_amount_in_usd) as total_funding FROM user_test_user.sample11111 WHERE last_funding_date IS NOT NULL GROUP BY last_funding_date ORDER BY last_funding_date',
    xField: 'last_funding_date',
    yField: 'total_funding',
    color: '#2e7d32',
  },
];

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

const BiReports: React.FC = () => {
  const [charts, setCharts] = useState<ChartConfig[]>(defaultCharts);
  const [chartData, setChartData] = useState<{ [key: string]: any[] }>({});
  const [loading, setLoading] = useState<{ [key: string]: boolean }>({});
  const [error, setError] = useState<{ [key: string]: string }>({});
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedChart, setSelectedChart] = useState<string | null>(null);
  const [customQuery, setCustomQuery] = useState('');
  const [showCustomQuery, setShowCustomQuery] = useState(false);

  const catalogService = new CatalogService();

  const fetchChartData = async (chart: ChartConfig) => {
    setLoading(prev => ({ ...prev, [chart.id]: true }));
    setError(prev => ({ ...prev, [chart.id]: '' }));

    try {
      const result = await catalogService.executeQuery(chart.query, 'test_user');
      if (result.status === 'success' && result.results) {
        const data = result.results.map((row: any[]) => ({
          [chart.xField]: row[0],
          [chart.yField]: parseFloat(row[1]) || 0,
        }));
        setChartData(prev => ({ ...prev, [chart.id]: data }));
      } else {
        setError(prev => ({ ...prev, [chart.id]: 'Failed to fetch data' }));
      }
    } catch (err) {
      setError(prev => ({ ...prev, [chart.id]: err instanceof Error ? err.message : 'An error occurred' }));
    } finally {
      setLoading(prev => ({ ...prev, [chart.id]: false }));
    }
  };

  useEffect(() => {
    charts.forEach(chart => fetchChartData(chart));
  }, [charts]);

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>, chartId: string) => {
    setAnchorEl(event.currentTarget);
    setSelectedChart(chartId);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedChart(null);
  };

  const handleRefresh = (chartId: string) => {
    const chart = charts.find(c => c.id === chartId);
    if (chart) {
      fetchChartData(chart);
    }
    handleMenuClose();
  };

  const handleCustomQuery = () => {
    setShowCustomQuery(true);
    handleMenuClose();
  };

  const handleCustomQuerySubmit = () => {
    if (selectedChart && customQuery) {
      const updatedCharts = charts.map(chart =>
        chart.id === selectedChart ? { ...chart, query: customQuery } : chart
      );
      setCharts(updatedCharts);
      setShowCustomQuery(false);
      setCustomQuery('');
    }
  };

  const renderChart = (chart: ChartConfig) => {
    const data = chartData[chart.id] || [];
    const isLoading = loading[chart.id];
    const errorMessage = error[chart.id];

    if (isLoading) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" height={300}>
          <CircularProgress />
        </Box>
      );
    }

    if (errorMessage) {
      return (
        <Box p={2}>
          <Alert severity="error">{errorMessage}</Alert>
        </Box>
      );
    }

    switch (chart.type) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={chart.xField} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey={chart.yField} fill={chart.color || COLORS[0]} />
            </BarChart>
          </ResponsiveContainer>
        );
      case 'line':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={chart.xField} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey={chart.yField}
                stroke={chart.color || COLORS[0]}
                activeDot={{ r: 8 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );
      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={data}
                dataKey={chart.yField}
                nameKey={chart.xField}
                cx="50%"
                cy="50%"
                outerRadius={100}
                label
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        );
      default:
        return null;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Business Intelligence Reports
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Interactive dashboards and charts for data analysis
      </Typography>

      {showCustomQuery && selectedChart && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Custom Query
          </Typography>
          <TextField
            fullWidth
            multiline
            rows={4}
            value={customQuery}
            onChange={(e) => setCustomQuery(e.target.value)}
            placeholder="Enter your SQL query..."
            sx={{ mb: 2 }}
          />
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button variant="contained" onClick={handleCustomQuerySubmit}>
              Apply Query
            </Button>
            <Button variant="outlined" onClick={() => setShowCustomQuery(false)}>
              Cancel
            </Button>
          </Box>
        </Paper>
      )}

      <Grid container spacing={3}>
        {charts.map((chart) => (
          <Grid item xs={12} md={6} key={chart.id}>
            <Card>
              <CardHeader
                title={chart.title}
                action={
                  <IconButton onClick={(e) => handleMenuClick(e, chart.id)}>
                    <MoreVertIcon />
                  </IconButton>
                }
              />
              <CardContent>
                {renderChart(chart)}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => handleRefresh(selectedChart!)}>
          <RefreshIcon sx={{ mr: 1 }} /> Refresh
        </MenuItem>
        <MenuItem onClick={handleCustomQuery}>
          Custom Query
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default BiReports; 