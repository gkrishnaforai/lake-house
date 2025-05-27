import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  CircularProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Grid,
  TextField,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Chip
} from '@mui/material';
import {
  BarChart as BarChartIcon,
  PieChart as PieChartIcon,
  ShowChart as LineChartIcon,
  TableChart as TableChartIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Save as SaveIcon
} from '@mui/icons-material';
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
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

interface SqlResultsProps {
  results: any;
  loading: boolean;
  error: string | null;
}

interface ChartConfig {
  id: string;
  type: 'bar' | 'line' | 'pie' | 'table';
  title: string;
  xAxisField: string;
  yAxisField: string;
  color?: string;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

export const SqlResults: React.FC<SqlResultsProps> = ({ results, loading, error }) => {
  const [charts, setCharts] = useState<ChartConfig[]>([]);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [editingChart, setEditingChart] = useState<ChartConfig | null>(null);
  const [reportTitle, setReportTitle] = useState('My Report');

  const columns = results?.columns || [];
  const data = results?.data || [];

  const handleAddChart = () => {
    setEditingChart({
      id: Date.now().toString(),
      type: 'bar',
      title: 'New Chart',
      xAxisField: columns[0] || '',
      yAxisField: columns[1] || ''
    });
    setConfigDialogOpen(true);
  };

  const handleEditChart = (chart: ChartConfig) => {
    setEditingChart(chart);
    setConfigDialogOpen(true);
  };

  const handleDeleteChart = (chartId: string) => {
    setCharts(charts.filter(chart => chart.id !== chartId));
  };

  const handleSaveChart = () => {
    if (!editingChart) return;

    setCharts(prevCharts => {
      const existingIndex = prevCharts.findIndex(chart => chart.id === editingChart.id);
      if (existingIndex >= 0) {
        const newCharts = [...prevCharts];
        newCharts[existingIndex] = editingChart;
        return newCharts;
      }
      return [...prevCharts, editingChart];
    });

    setConfigDialogOpen(false);
    setEditingChart(null);
  };

  const renderChart = (chart: ChartConfig) => {
    if (!data.length) return null;

    switch (chart.type) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={chart.xAxisField} />
              <YAxis />
              <RechartsTooltip />
              <Legend />
              <Bar dataKey={chart.yAxisField} fill={chart.color || COLORS[0]} />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'line':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={chart.xAxisField} />
              <YAxis />
              <RechartsTooltip />
              <Legend />
              <Line type="monotone" dataKey={chart.yAxisField} stroke={chart.color || COLORS[0]} />
            </LineChart>
          </ResponsiveContainer>
        );

      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={data}
                dataKey={chart.yAxisField}
                nameKey={chart.xAxisField}
                cx="50%"
                cy="50%"
                outerRadius={100}
                label
              >
                {data.map((entry: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <RechartsTooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        );

      case 'table':
        return (
          <TableContainer component={Paper}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  {columns.map((column: string) => (
                    <TableCell key={column}>{column}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {data.map((row: any, rowIndex: number) => (
                  <TableRow key={rowIndex}>
                    {columns.map((column: string) => (
                      <TableCell key={`${rowIndex}-${column}`}>{row[column]}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        );

      default:
        return null;
    }
  };

  return (
    <Card sx={{ mt: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <TextField
            label="Report Title"
            value={reportTitle}
            onChange={(e) => setReportTitle(e.target.value)}
            variant="outlined"
            size="small"
          />
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleAddChart}
          >
            Add Visualization
          </Button>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {loading ? (
          <Box display="flex" justifyContent="center" p={2}>
            <CircularProgress />
          </Box>
        ) : (
          <Grid container spacing={2}>
            {charts.map((chart) => (
              <Grid item xs={12} md={6} key={chart.id}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h6">{chart.title}</Typography>
                      <Box>
                        <Tooltip title="Edit">
                          <IconButton size="small" onClick={() => handleEditChart(chart)}>
                            <EditIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton size="small" onClick={() => handleDeleteChart(chart.id)}>
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>
                    {renderChart(chart)}
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </CardContent>

      <Dialog
        open={configDialogOpen}
        onClose={() => {
          setConfigDialogOpen(false);
          setEditingChart(null);
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {editingChart?.id ? 'Edit Visualization' : 'Add Visualization'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              fullWidth
              label="Chart Title"
              value={editingChart?.title || ''}
              onChange={(e) => setEditingChart(prev => prev ? { ...prev, title: e.target.value } : null)}
            />

            <FormControl fullWidth>
              <InputLabel>Chart Type</InputLabel>
              <Select
                value={editingChart?.type || 'bar'}
                onChange={(e) => setEditingChart(prev => prev ? { ...prev, type: e.target.value as ChartConfig['type'] } : null)}
                label="Chart Type"
              >
                <MenuItem value="bar">Bar Chart</MenuItem>
                <MenuItem value="line">Line Chart</MenuItem>
                <MenuItem value="pie">Pie Chart</MenuItem>
                <MenuItem value="table">Table</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel>X-Axis Field</InputLabel>
              <Select
                value={editingChart?.xAxisField || ''}
                onChange={(e) => setEditingChart(prev => prev ? { ...prev, xAxisField: e.target.value } : null)}
                label="X-Axis Field"
              >
                {columns.map((column: string) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel>Y-Axis Field</InputLabel>
              <Select
                value={editingChart?.yAxisField || ''}
                onChange={(e) => setEditingChart(prev => prev ? { ...prev, yAxisField: e.target.value } : null)}
                label="Y-Axis Field"
              >
                {columns.map((column: string) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel>Color</InputLabel>
              <Select
                value={editingChart?.color || COLORS[0]}
                onChange={(e) => setEditingChart(prev => prev ? { ...prev, color: e.target.value } : null)}
                label="Color"
              >
                {COLORS.map((color) => (
                  <MenuItem key={color} value={color}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box
                        sx={{
                          width: 20,
                          height: 20,
                          backgroundColor: color,
                          borderRadius: 1
                        }}
                      />
                      {color}
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setConfigDialogOpen(false);
            setEditingChart(null);
          }}>
            Cancel
          </Button>
          <Button
            onClick={handleSaveChart}
            variant="contained"
            startIcon={<SaveIcon />}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
}; 