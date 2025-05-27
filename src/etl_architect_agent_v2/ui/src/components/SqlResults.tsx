import React, { useState, useCallback, useMemo } from 'react';
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
  Chip,
  Menu
} from '@mui/material';
import {
  BarChart as BarChartIcon,
  PieChart as PieChartIcon,
  ShowChart as LineChartIcon,
  TableChart as TableChartIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  FilterList as FilterIcon,
  Sort as SortIcon,
  Functions as FunctionsIcon,
  Download as DownloadIcon
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
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import { ColDef, GridApi, GridReadyEvent, ValueFormatterParams, IRowNode } from 'ag-grid-community';
import * as XLSX from 'xlsx';

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

interface Formula {
  id: string;
  name: string;
  formula: string;
  description: string;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

export const SqlResults: React.FC<SqlResultsProps> = ({ results, loading, error }) => {
  const [charts, setCharts] = useState<ChartConfig[]>([]);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [editingChart, setEditingChart] = useState<ChartConfig | null>(null);
  const [reportTitle, setReportTitle] = useState('My Report');
  const [gridApi, setGridApi] = useState<GridApi | null>(null);
  const [formulaMenuAnchor, setFormulaMenuAnchor] = useState<null | HTMLElement>(null);
  const [formulaDialogOpen, setFormulaDialogOpen] = useState(false);
  const [selectedFormula, setSelectedFormula] = useState<Formula | null>(null);
  const [columnMenuAnchor, setColumnMenuAnchor] = useState<null | HTMLElement>(null);
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null);

  const columns = results?.columns || [];
  const data = results?.data || [];

  const defaultColDef = useMemo(() => ({
    sortable: true,
    filter: true,
    resizable: true,
    editable: true,
    flex: 1,
    minWidth: 100,
  }), []);

  const columnDefs = useMemo(() => {
    return columns.map((col: string) => ({
      field: col,
      headerName: col,
      valueFormatter: (params: ValueFormatterParams) => {
        if (params.value === null || params.value === undefined) return '';
        return params.value.toString();
      }
    }));
  }, [columns]);

  const onGridReady = useCallback((params: GridReadyEvent) => {
    setGridApi(params.api);
  }, []);

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

  const handleExportToExcel = () => {
    if (!gridApi) return;

    const rowData: any[] = [];
    gridApi.forEachNode((rowNode: IRowNode<any>) => {
      if (rowNode.data) {
        rowData.push(rowNode.data);
      }
    });

    const worksheet = XLSX.utils.json_to_sheet(rowData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Query Results');
    XLSX.writeFile(workbook, 'query_results.xlsx');
  };

  const handleFormulaClick = (event: React.MouseEvent<HTMLElement>) => {
    setFormulaMenuAnchor(event.currentTarget);
  };

  const handleFormulaClose = () => {
    setFormulaMenuAnchor(null);
  };

  const handleFormulaSelect = (formula: Formula) => {
    setSelectedFormula(formula);
    setFormulaDialogOpen(true);
    handleFormulaClose();
  };

  const handleColumnMenuClick = (event: React.MouseEvent<HTMLElement>, column: string) => {
    setSelectedColumn(column);
    setColumnMenuAnchor(event.currentTarget);
  };

  const handleColumnMenuClose = () => {
    setColumnMenuAnchor(null);
    setSelectedColumn(null);
  };

  const handleColumnOperation = (operation: string) => {
    if (!gridApi || !selectedColumn) return;

    let result: number;
    const values: number[] = [];

    gridApi.forEachNode((rowNode: IRowNode<any>) => {
      if (rowNode.data && rowNode.data[selectedColumn] !== null && rowNode.data[selectedColumn] !== undefined) {
        const value = parseFloat(rowNode.data[selectedColumn]);
        if (!isNaN(value)) {
          values.push(value);
        }
      }
    });

    switch (operation) {
      case 'sum':
        result = values.reduce((a, b) => a + b, 0);
        break;
      case 'average':
        result = values.reduce((a, b) => a + b, 0) / values.length;
        break;
      case 'min':
        result = Math.min(...values);
        break;
      case 'max':
        result = Math.max(...values);
        break;
      default:
        return;
    }

    // Add result as a new row
    const newRow = { [selectedColumn]: result };
    gridApi.applyTransaction({ add: [newRow] });
    handleColumnMenuClose();
  };

  const predefinedFormulas: Formula[] = [
    {
      id: '1',
      name: 'Sum',
      formula: '=SUM({column})',
      description: 'Calculate the sum of values in a column'
    },
    {
      id: '2',
      name: 'Average',
      formula: '=AVG({column})',
      description: 'Calculate the average of values in a column'
    },
    {
      id: '3',
      name: 'Count',
      formula: '=COUNT({column})',
      description: 'Count the number of non-null values in a column'
    },
    {
      id: '4',
      name: 'Min',
      formula: '=MIN({column})',
      description: 'Find the minimum value in a column'
    },
    {
      id: '5',
      name: 'Max',
      formula: '=MAX({column})',
      description: 'Find the maximum value in a column'
    }
  ];

  return (
    <Card sx={{ mt: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Query Results</Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title="Export to Excel">
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={handleExportToExcel}
                size="small"
              >
                Export to Excel
              </Button>
            </Tooltip>
            <Tooltip title="Add Formula">
              <Button
                variant="outlined"
                startIcon={<FunctionsIcon />}
                onClick={handleFormulaClick}
                size="small"
              >
                Formulas
              </Button>
            </Tooltip>
          </Box>
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
          <>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                Click column headers to sort, drag to resize columns, and double-click cells to edit
              </Typography>
            </Box>
            <div className="ag-theme-alpine" style={{ height: 500, width: '100%' }}>
              <AgGridReact
                columnDefs={columnDefs}
                rowData={data}
                defaultColDef={defaultColDef}
                onGridReady={onGridReady}
                enableRangeSelection={true}
                copyHeadersToClipboard={true}
                suppressRowClickSelection={true}
                rowSelection="multiple"
                pagination={true}
                paginationPageSize={100}
              />
            </div>
          </>
        )}

        <Menu
          anchorEl={formulaMenuAnchor}
          open={Boolean(formulaMenuAnchor)}
          onClose={handleFormulaClose}
        >
          {predefinedFormulas.map((formula) => (
            <MenuItem
              key={formula.id}
              onClick={() => handleFormulaSelect(formula)}
            >
              <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                <Typography variant="body1">{formula.name}</Typography>
                <Typography variant="caption" color="text.secondary">
                  {formula.description}
                </Typography>
              </Box>
            </MenuItem>
          ))}
        </Menu>

        <Menu
          anchorEl={columnMenuAnchor}
          open={Boolean(columnMenuAnchor)}
          onClose={handleColumnMenuClose}
        >
          <MenuItem onClick={() => handleColumnOperation('sum')}>
            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
              <Typography variant="body1">Sum</Typography>
              <Typography variant="caption" color="text.secondary">
                Calculate the sum of values
              </Typography>
            </Box>
          </MenuItem>
          <MenuItem onClick={() => handleColumnOperation('average')}>
            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
              <Typography variant="body1">Average</Typography>
              <Typography variant="caption" color="text.secondary">
                Calculate the average of values
              </Typography>
            </Box>
          </MenuItem>
          <MenuItem onClick={() => handleColumnOperation('min')}>
            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
              <Typography variant="body1">Minimum</Typography>
              <Typography variant="caption" color="text.secondary">
                Find the minimum value
              </Typography>
            </Box>
          </MenuItem>
          <MenuItem onClick={() => handleColumnOperation('max')}>
            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
              <Typography variant="body1">Maximum</Typography>
              <Typography variant="caption" color="text.secondary">
                Find the maximum value
              </Typography>
            </Box>
          </MenuItem>
        </Menu>

        <Dialog
          open={formulaDialogOpen}
          onClose={() => setFormulaDialogOpen(false)}
          maxWidth="sm"
          fullWidth
        >
          <DialogTitle>Add Formula</DialogTitle>
          <DialogContent>
            {selectedFormula && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle1">{selectedFormula.name}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {selectedFormula.description}
                </Typography>
                <TextField
                  fullWidth
                  label="Formula"
                  value={selectedFormula.formula}
                  margin="normal"
                  InputProps={{ readOnly: true }}
                />
              </Box>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setFormulaDialogOpen(false)}>Cancel</Button>
            <Button variant="contained" onClick={() => setFormulaDialogOpen(false)}>
              Apply
            </Button>
          </DialogActions>
        </Dialog>
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