import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Chip,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  CardActions,
  IconButton,
  Tooltip,
  ToggleButton,
  ToggleButtonGroup,
  Stack,
  Tabs,
  Tab,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  ListItem,
  Divider,
  List,
  FormControl,
  InputLabel,
  Select,
  SelectChangeEvent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar
} from '@mui/material';
import {
  Search as SearchIcon,
  Download as DownloadIcon,
  FilterList as FilterIcon,
  Sort as SortIcon,
  Refresh as RefreshIcon,
  TableRows as TableIcon,
  BarChart as BarChartIcon,
  ShowChart as LineChartIcon,
  PieChart as PieChartIcon,
  ScatterPlot as ScatterPlotIcon,
  BubbleChart as BubbleChartIcon,
  ShowChart as ShowChartIcon,
  Add as AddIcon,
  Remove as RemoveIcon,
  Assessment as ReportIcon,
  Save as SaveIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Share as ShareIcon,
  Schedule as ScheduleIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  PlayArrow as PlayArrowIcon,
  Functions as FunctionsIcon
} from '@mui/icons-material';
import { CatalogService, getUserId } from '../services/catalogService';
import { TableInfo, SavedQuery } from '../types/api';
import { TransformationTab } from './TransformationTab';
import TabPanel from './TabPanel';
import FileUpload from './FileUpload';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';
import * as XLSX from 'xlsx';
import { reportService, Report, ReportInput } from '../services/reportService';

interface QueryResult {
  status: string;
  results?: any[][];
  query?: string;
  message?: string;
  columns_used?: string[];
  sql_query?: string;
  explanation?: string;
  confidence?: number;
  tables_used?: string[];
  filters?: Record<string, any>;
  error?: string | null;
}

interface DescriptiveQueryResult {
  status: string;
  results?: any[];
  query?: string;
  message?: string;
  columns_used?: string[];
  sql_query?: string;
  explanation?: string;
  confidence?: number;
  tables_used?: string[];
  filters?: Record<string, any>;
  error?: string | null;
}

interface QueryHistoryItem {
  query: string;
  sql: string;
  timestamp: string;
}

interface DataExplorerProps {
  selectedTables: TableInfo[];
  onTableSelect: (table: TableInfo) => void;
  onQueryExecute: (query: string) => void;
}

type ChartType = 'bar' | 'line' | 'pie' | 'scatter' | 'bubble';
type AggregationType = 'count' | 'sum' | 'avg' | 'min' | 'max';

interface ChartConfig {
  id: string;
  type: 'bar' | 'line' | 'pie' | 'scatter' | 'bubble';
  title: string;
  xAxis: string;
  yAxis: string[];
  groupBy?: string;
  aggregation?: 'sum' | 'average' | 'count' | 'min' | 'max';
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

const DataExplorer: React.FC<DataExplorerProps> = ({ selectedTables, onTableSelect, onQueryExecute }) => {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'sql' | 'descriptive'>('sql');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [data, setData] = useState<any[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [schema, setSchema] = useState<{ name: string; type: string; comment: string }[]>([]);
  const [schemaCache, setSchemaCache] = useState<Record<string, { name: string; type: string; comment: string }[]>>({});
  const [isSchemaTabActive, setIsSchemaTabActive] = useState(false);
  const [showSchemaAlert, setShowSchemaAlert] = useState(false);
  const [queryHistory, setQueryHistory] = useState<QueryHistoryItem[]>([]);
  const [generatedSql, setGeneratedSql] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(0);
  const [selectedTable, setSelectedTable] = useState<TableInfo | null>(null);
  const [contextMenu, setContextMenu] = useState<{
    mouseX: number;
    mouseY: number;
    table: TableInfo | null;
  } | null>(null);
  const [chartConfig, setChartConfig] = useState<ChartConfig>({
    id: '',
    type: 'bar',
    title: '',
    xAxis: '',
    yAxis: [],
  });
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [reports, setReports] = useState<Report[]>([]);
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [reportDialogOpen, setReportDialogOpen] = useState(false);
  const [reportForm, setReportForm] = useState<{
    name: string;
    description: string;
    query: string;
    schedule: string;
    isFavorite: boolean;
  }>({
    name: '',
    description: '',
    query: '',
    schedule: '',
    isFavorite: false
  });
  const [reportDialogMode, setReportDialogMode] = useState<'create' | 'edit'>('create');
  const [editingReportId, setEditingReportId] = useState<string | null>(null);
  const [formulaMenuAnchor, setFormulaMenuAnchor] = useState<null | HTMLElement>(null);
  const [columnMenuAnchor, setColumnMenuAnchor] = useState<null | HTMLElement>(null);
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null);
  const [sortConfig, setSortConfig] = useState<{ column: string; direction: 'asc' | 'desc' } | null>(null);
  const [formulaResults, setFormulaResults] = useState<Record<string, number>>({});
  const [charts, setCharts] = useState<ChartConfig[]>([]);
  const [chartDialogOpen, setChartDialogOpen] = useState(false);
  const [editingChart, setEditingChart] = useState<ChartConfig | null>(null);
  const [loadingReports, setLoadingReports] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);
  const [saveQueryDialogOpen, setSaveQueryDialogOpen] = useState(false);
  const [saveQueryForm, setSaveQueryForm] = useState({
    name: '',
    description: '',
  });
  const [savedQueries, setSavedQueries] = useState<SavedQuery[]>([]);
  const [selectedQuery, setSelectedQuery] = useState<SavedQuery | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const catalogService = new CatalogService();

  const handleTableClick = (table: TableInfo) => {
    console.log('Table clicked:', table.name);
    setSelectedTable(table);
    onTableSelect(table);
  };

  const handleContextMenuClose = () => {
    setContextMenu(null);
  };

  const handleContextMenu = (event: React.MouseEvent, table: TableInfo) => {
    event.preventDefault();
    setContextMenu({
      mouseX: event.clientX - 2,
      mouseY: event.clientY - 4,
      table
    });
  };

  const handleExport = async (table: TableInfo) => {
    try {
      const response = await fetch(`/api/catalog/tables/${table.name}/export`, {
        method: 'GET',
        headers: {
          'Accept': 'text/csv'
        }
      });
      
      if (!response.ok) {
        throw new Error('Export failed');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${table.name}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting table:', error);
      // You might want to show an error notification here
    }
    handleContextMenuClose();
  };

  const fetchSchema = async (tableName: string) => {
    try {
      // Check if schema is already in cache
      if (schemaCache[tableName]) {
        console.log('Using cached schema for table:', tableName);
        setSchema(schemaCache[tableName]);
        setShowSchemaAlert(true);
        return;
      }

      console.log('Fetching schema for table:', tableName);
      const data = await catalogService.getTableSchema(tableName);
      console.log('Raw schema data received:', data);
      
      if (data && Array.isArray(data.schema)) {
        const mappedSchema = data.schema.map((col: any) => ({
          name: col.name,
          type: col.type,
          comment: col.comment
        }));
        console.log('Mapped schema before setState:', mappedSchema);
        
        // Update both schema state and cache
        setSchema(mappedSchema);
        setSchemaCache(prev => ({
          ...prev,
          [tableName]: mappedSchema
        }));
        setShowSchemaAlert(true);
        console.log('Schema state and cache updated');
      } else {
        console.log('Invalid schema data format:', data);
        setSchema([]);
        setShowSchemaAlert(false);
      }
    } catch (error) {
      console.error('Error in fetchSchema:', error);
      setSchema([]);
      setShowSchemaAlert(false);
    }
  };

  const handleModeChange = (_: any, newMode: 'sql' | 'descriptive') => {
    if (newMode) setMode(newMode);
    setQuery('');
    setData([]);
    setColumns([]);
    setError(null);
    setDownloadUrl(null);
  };

  const handleSchemaTabClick = () => {
    console.log('Schema tab clicked');
    setIsSchemaTabActive(true);
    if (selectedTables.length > 0) {
      const table = selectedTables[0];
      console.log('Fetching schema for table:', table.name);
      fetchSchema(table.name);
    }
  };

  useEffect(() => {
    console.log('useEffect triggered with selectedTables:', selectedTables);
    if (selectedTables.length > 0) {
      const table = selectedTables[0];
      console.log('Table selected:', table.name);
      setSelectedTable(table);
      fetchSchema(table.name);
    } else {
      console.log('No tables selected, clearing schema');
      setSelectedTable(null);
      setSchema([]);
      setShowSchemaAlert(false);
    }
  }, [selectedTables]);

  useEffect(() => {
    console.log('Schema state changed:', schema);
  }, [schema]);

  const handleQuerySubmit = async () => {
    console.log('Submitting query with selectedTable:', selectedTable);
    if (!selectedTable) {
      setError("Please select a table first");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await catalogService.descriptiveQuery(
        query,
        selectedTable.name,
        "true",
        "test_user"
      );
      
      if (result.status === "success") {
        console.log('Query result:', result);
        
        if (result.results && Array.isArray(result.results) && result.results.length > 0) {
          if (result.metadata?.columns_used) {
            console.log('Using columns from metadata:', result.metadata.columns_used);
            setColumns(result.metadata.columns_used);
          } else if (result.columns && Array.isArray(result.columns)) {
            console.log('Using columns from result:', result.columns);
            setColumns(result.columns);
          } else {
            const firstRow = result.results[0];
            const inferredColumns = Object.keys(firstRow);
            console.log('Inferred columns from first row:', inferredColumns);
            setColumns(inferredColumns);
          }
          
          const dataObjects = result.results.map((row: any) => {
            if (typeof row === 'object' && row !== null) {
              return row;
            } else if (Array.isArray(row)) {
              const obj: any = {};
              const headers = result.metadata?.columns_used || result.columns || 
                            Array.from({ length: row.length }, (_, i) => `Column ${i + 1}`);
              headers.forEach((col: string, i: number) => {
                obj[col] = row[i];
              });
              return obj;
            }
            return row;
          });
          
          console.log('Processed data objects:', dataObjects);
          setData(dataObjects);
          setGeneratedSql(result.query);
          setQueryHistory(prev => [...prev, {
            query,
            sql: result.query || '',
            timestamp: new Date().toISOString()
          }]);
        } else {
          setError('Info - No results returned from query');
        }
      } else {
        setError(result.message || "Query failed");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    console.log('Data state updated:', data);
    console.log('Columns state updated:', columns);
    if (data.length > 0 && columns.length > 0) {
      //setError(`Info - Data grid should be visible with ${data.length} rows and ${columns.length} columns`);
    } else {
      setError(`Info - Data grid not visible. Data length: ${data.length}, Columns length: ${columns.length}`);
    }
  }, [data, columns]);

  useEffect(() => {
    console.log('Selected tables changed:', selectedTables);
    if (selectedTables.length > 0) {
      setError(null);
    } else {
      setError('Info - No table selected');
    }
  }, [selectedTables]);

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleUploadSuccess = () => {
    console.log('Upload successful, refreshing schema...');
    if (selectedTables.length > 0) {
      fetchSchema(selectedTables[0].name);
    }
  };

  const handleUploadError = (error: string) => {
    console.error('Upload error:', error);
    setError(error);
  };

  const handleChartTypeChange = (type: ChartType) => {
    setChartConfig(prev => ({
      ...prev,
      type: type,
    }));
  };

  const handleXAxisChange = (event: SelectChangeEvent) => {
    setChartConfig(prev => ({
      ...prev,
      xAxis: event.target.value,
    }));
  };

  const handleYAxisChange = (event: SelectChangeEvent) => {
    setChartConfig(prev => ({
      ...prev,
      yAxis: [event.target.value],
    }));
  };

  const handleGroupByChange = (event: SelectChangeEvent) => {
    setChartConfig(prev => ({
      ...prev,
      groupBy: event.target.value,
    }));
  };

  const handleAggregationChange = (event: SelectChangeEvent) => {
    setChartConfig(prev => ({
      ...prev,
      aggregation: event.target.value as ChartConfig['aggregation'],
    }));
  };

  const handleSortChange = (event: SelectChangeEvent) => {
    const [column, direction] = event.target.value.split(':');
    setChartConfig(prev => ({
      ...prev,
      sortBy: { column, direction: direction as 'asc' | 'desc' },
    }));
  };

  const generateQuery = () => {
    const table = selectedTables[0];
    if (!table) return '';

    let query = `SELECT `;
    
    query += `${chartConfig.xAxis}, `;
    
    query += chartConfig.yAxis.map(y => 
      `${chartConfig.aggregation || 'sum'}(${y}) as ${y}`
    ).join(', ');
    
    query += ` FROM ${table.name}`;
    
    if (chartConfig.groupBy) {
      query += ` GROUP BY ${chartConfig.xAxis}`;
    }
    
    return query;
  };

  const fetchData = async () => {
    if (selectedTables.length === 0) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const query = generateQuery();
      console.log('Generated query:', query);
      
      const result = await catalogService.sqlQuery(
        query,
        [selectedTables[0].name],
        'true'
      );
      
      setData(result);
      setPreviewData(result.slice(0, 5));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  const renderChart = (chart: ChartConfig) => {
    if (!data.length) return null;

    const chartData = data.map(row => {
      const dataPoint: any = {};
      dataPoint[chart.xAxis] = Array.isArray(row) ? row[columns.indexOf(chart.xAxis)] : row[chart.xAxis];
      chart.yAxis.forEach(yAxis => {
        dataPoint[yAxis] = Array.isArray(row) ? row[columns.indexOf(yAxis)] : row[yAxis];
      });
      return dataPoint;
    });

    switch (chart.type) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={chart.xAxis} />
              <YAxis />
              <RechartsTooltip />
              <Legend />
              {chart.yAxis.map((y, index) => (
                <Bar
                  key={y}
                  dataKey={y}
                  fill={COLORS[index % COLORS.length]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        );

      case 'line':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={chart.xAxis} />
              <YAxis />
              <RechartsTooltip />
              <Legend />
              {chart.yAxis.map((y, index) => (
                <Line
                  key={y}
                  type="monotone"
                  dataKey={y}
                  stroke={COLORS[index % COLORS.length]}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        );

      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={chartData}
                dataKey={chart.yAxis[0]}
                nameKey={chart.xAxis}
                cx="50%"
                cy="50%"
                outerRadius={100}
                label
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <RechartsTooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        );

      case 'scatter':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={chart.xAxis} />
              <YAxis dataKey={chart.yAxis[0]} />
              <RechartsTooltip />
              <Legend />
              <Scatter
                data={chartData}
                fill={COLORS[0]}
              />
            </ScatterChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  const handleEditReport = (report: Report) => {
    if (!report.id) {
      setError('Cannot edit report: missing report ID');
      return;
    }
    setReportDialogMode('edit');
    setEditingReportId(report.id);
    setReportForm({
      name: report.name,
      description: report.description,
      query: report.query,
      schedule: report.schedule || '',
      isFavorite: report.isFavorite || false
    });
    setReportDialogOpen(true);
  };

  const handleSaveReport = async () => {
    if (!reportForm.name || !reportForm.description || !reportForm.query) {
      setError('Please fill in all required fields');
      return;
    }

    try {
      const reportInput: ReportInput = {
        name: reportForm.name,
        description: reportForm.description,
        query: reportForm.query,
        schedule: reportForm.schedule,
        isFavorite: reportForm.isFavorite
      };

      if (reportDialogMode === 'edit' && editingReportId) {
        await reportService.updateReport(editingReportId, reportInput);
      } else {
        await reportService.saveReport(reportInput);
      }

      await loadReports();
      setReportDialogOpen(false);
      setReportForm({
        name: '',
        description: '',
        query: '',
        schedule: '',
        isFavorite: false
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save report');
    }
  };

  const handleCloseReportDialog = () => {
    setReportDialogOpen(false);
    setReportForm({
      name: '',
      description: '',
      query: '',
      schedule: '',
      isFavorite: false
    });
    setReportDialogMode('create');
    setEditingReportId(null);
  };

  const handleDeleteReport = async (report: Report) => {
    if (!report.id) {
      setError('Cannot delete report: missing report ID');
      return;
    }
    try {
      await reportService.deleteReport(report.id);
      await loadReports();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete report');
    }
  };

  const handleToggleFavorite = (reportId: string) => {
    setReports(prev => prev.map(r => 
      r.id === reportId ? { ...r, isFavorite: !r.isFavorite } : r
    ));
  };

  const handleRunReport = async (report: Report) => {
    if (!report.id) {
      setError('Cannot run report: missing report ID');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await reportService.runReport(report.id);
      setData(result.results || []);
      setColumns(result.columns || []);
      setGeneratedSql(report.query);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run report');
    } finally {
      setLoading(false);
    }
  };

  const handleScheduleReport = async (report: Report) => {
    if (!report.id) {
      setError('Cannot schedule report: missing report ID');
      return;
    }
    try {
      await reportService.scheduleReport(report.id, report.schedule || '');
      await loadReports();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to schedule report');
    }
  };

  const handleShareReport = async (report: Report) => {
    try {
      const response = await fetch('/api/catalog/reports/share', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          report_id: report.id,
          report_name: report.name,
          query: report.query,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to share report');
      }

      const result = await response.json();
      console.log('Report shared:', result);
      
    } catch (err) {
      console.error('Error sharing report:', err);
      setError(err instanceof Error ? err.message : 'Failed to share report');
    }
  };

  const handleExportToExcel = () => {
    const worksheet = XLSX.utils.json_to_sheet(data);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Data Explorer');
    XLSX.writeFile(workbook, 'data_explorer.xlsx');
  };

  const handleFormulaClick = (event: React.MouseEvent<HTMLElement>) => {
    setFormulaMenuAnchor(event.currentTarget);
  };

  const handleFormulaClose = () => {
    setFormulaMenuAnchor(null);
  };

  const handleColumnMenuClick = (event: React.MouseEvent<HTMLElement>, column: string) => {
    setSelectedColumn(column);
    setColumnMenuAnchor(event.currentTarget);
  };

  const handleColumnMenuClose = () => {
    setColumnMenuAnchor(null);
    setSelectedColumn(null);
  };

  const handleSort = (column: string) => {
    setSortConfig(prev => ({
      column,
      direction: prev?.column === column && prev.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const handleColumnOperation = (operation: string) => {
    if (!selectedColumn) return;

    let result: number;
    const values: number[] = data
      .map(row => {
        const value = Array.isArray(row) ? row[columns.indexOf(selectedColumn)] : row[selectedColumn];
        return typeof value === 'number' ? value : parseFloat(String(value));
      })
      .filter(value => !isNaN(value));

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

    // Update formula results
    setFormulaResults(prev => ({
      ...prev,
      [`${selectedColumn}_${operation}`]: result
    }));
    handleColumnMenuClose();
  };

  const getSortedData = () => {
    if (!sortConfig) return data;

    return [...data].sort((a, b) => {
      const aValue = Array.isArray(a) ? a[columns.indexOf(sortConfig.column)] : a[sortConfig.column];
      const bValue = Array.isArray(b) ? b[columns.indexOf(sortConfig.column)] : b[sortConfig.column];

      if (aValue === bValue) return 0;
      if (aValue === null || aValue === undefined) return 1;
      if (bValue === null || bValue === undefined) return -1;

      const comparison = aValue < bValue ? -1 : 1;
      return sortConfig.direction === 'asc' ? comparison : -comparison;
    });
  };

  const handleAddChart = () => {
    setEditingChart({
      id: Date.now().toString(),
      type: 'bar',
      title: 'New Chart',
      xAxis: columns[0] || '',
      yAxis: [columns[1] || ''],
    });
    setChartDialogOpen(true);
  };

  const handleEditChart = (chart: ChartConfig) => {
    setEditingChart(chart);
    setChartDialogOpen(true);
  };

  const handleDeleteChart = (chartId: string) => {
    setCharts(charts.filter(chart => chart.id !== chartId));
  };

  const handleSaveChart = () => {
    if (!editingChart) return;

    setCharts(prev => {
      const existingIndex = prev.findIndex(chart => chart.id === editingChart.id);
      if (existingIndex >= 0) {
        const newCharts = [...prev];
        newCharts[existingIndex] = editingChart;
        return newCharts;
      }
      return [...prev, editingChart];
    });

    setChartDialogOpen(false);
    setEditingChart(null);
  };

  const loadReports = async () => {
    setLoadingReports(true);
    setReportError(null);
    try {
      const userReports = await reportService.getReports();
      setReports(userReports);
    } catch (error) {
      setReportError(error instanceof Error ? error.message : 'Failed to load reports');
    } finally {
      setLoadingReports(false);
    }
  };

  const handleSaveQuery = async () => {
    if (!saveQueryForm.name || !query) {
      setError('Query name and SQL query are required');
      return;
    }

    try {
      const newReport: ReportInput = {
        name: saveQueryForm.name,
        description: saveQueryForm.description || '',
        query: query,
        isFavorite: false
      };

      await reportService.saveReport(newReport);
      await loadReports();
      setSaveQueryDialogOpen(false);
      setSaveQueryForm({
        name: '',
        description: '',
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save query');
    }
  };

  useEffect(() => {
    loadSavedQueries();
  }, []);

  const loadSavedQueries = async () => {
    try {
      const userId = getUserId();
      const queries = await catalogService.getSavedQueries(userId);
      setSavedQueries(queries);
    } catch (error) {
      setError('Failed to load saved queries');
    }
  };

  const handleDeleteQuery = async (queryId: string) => {
    try {
      const userId = getUserId();
      await catalogService.deleteQuery(userId, queryId);
      setSavedQueries(savedQueries.filter(q => q.query_id !== queryId));
      setSuccess('Query deleted successfully');
    } catch (error) {
      setError('Failed to delete query');
    }
  };

  const handleToggleFavoriteQuery = async (queryId: string, isFavorite: boolean) => {
    try {
      const userId = getUserId();
      await catalogService.updateQueryFavorite(userId, queryId, !isFavorite);
      setSavedQueries(savedQueries.map(q => 
        q.query_id === queryId ? { ...q, is_favorite: !isFavorite } : q
      ));
    } catch (error) {
      setError('Failed to update favorite status');
    }
  };

  const handleSelectQuery = (savedQuery: SavedQuery) => {
    setQuery(savedQuery.query);
    setSelectedQuery(savedQuery);
  };

  const handleExecuteQuery = async (savedQuery: SavedQuery) => {
    try {
      const userId = getUserId();
      await catalogService.updateQueryExecution(userId, savedQuery.query_id);
      onQueryExecute(savedQuery.query);
    } catch (error) {
      setError('Failed to execute query');
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={(_, newValue) => {
          console.log('Tab changed to:', newValue);
          setActiveTab(newValue);
        }}>
          <Tab label="Data" />
          <Tab label="Schema" />
          <Tab label="Transformations" />
          <Tab label="Upload" />
        </Tabs>
      </Box>

      <Menu
        open={contextMenu !== null}
        onClose={handleContextMenuClose}
        anchorReference="anchorPosition"
        anchorPosition={
          contextMenu !== null
            ? { top: contextMenu.mouseY, left: contextMenu.mouseX }
            : undefined
        }
      >
        <MenuItem onClick={() => contextMenu?.table && handleExport(contextMenu.table)}>
          <ListItemIcon>
            <DownloadIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Export as CSV</ListItemText>
        </MenuItem>
      </Menu>

      <TabPanel value={activeTab} index={0}>
        <Box sx={{ p: 2 }}>
          {selectedTables.length > 0 && (
            <Alert severity="info" sx={{ mb: 2 }}>
              Selected table: {selectedTables[0].name}
            </Alert>
          )}

          <Paper sx={{ p: 2, mb: 2 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Available Tables
                </Typography>
                <List>
                  {selectedTables.map((table) => (
                    <React.Fragment key={table.name}>
                      <ListItem 
                        button 
                        selected={selectedTable?.name === table.name}
                        onClick={() => handleTableClick(table)}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          setContextMenu({
                            mouseX: e.clientX - 2,
                            mouseY: e.clientY - 4,
                            table
                          });
                        }}
                      >
                        <ListItemIcon>
                          <TableIcon />
                        </ListItemIcon>
                        <ListItemText
                          primary={table.name}
                          secondary={table.description || 'No description'}
                        />
                        <Chip 
                          label={`${table.rowCount?.toLocaleString() || 'N/A'} rows`} 
                          color="primary" 
                          size="small" 
                        />
                      </ListItem>
                      <Divider />
                    </React.Fragment>
                  ))}
                </List>
              </Grid>
              <Grid item xs={12}>
                <ToggleButtonGroup
                  value={mode}
                  exclusive
                  onChange={handleModeChange}
                  size="small"
                  sx={{ mb: 2 }}
                >
                  <ToggleButton value="sql">SQL Query</ToggleButton>
                  <ToggleButton value="descriptive">Chat with Data</ToggleButton>
                </ToggleButtonGroup>
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  multiline
                  rows={4}
                  variant="outlined"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={mode === 'sql' ? 'Enter your SQL query...' : 'Describe what you want to know about the data...'}
                  sx={{ mb: 2 }}
                />
              </Grid>
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                  <Button
                    variant="contained"
                    onClick={handleQuerySubmit}
                    disabled={loading || !query.trim()}
                    startIcon={loading ? <CircularProgress size={20} /> : <SearchIcon />}
                  >
                    {loading ? 'Running...' : 'Run Query'}
                  </Button>
                  
                </Box>
              </Grid>
            </Grid>
          </Paper>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {generatedSql && (
            <Paper sx={{ p: 2, mb: 2, bgcolor: 'grey.100' }}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Generated SQL:
              </Typography>
              <Typography
                component="pre"
                sx={{
                  p: 1,
                  bgcolor: 'white',
                  borderRadius: 1,
                  overflowX: 'auto',
                  fontSize: '0.875rem',
                  fontFamily: 'monospace'
                }}
              >
                {generatedSql}
              </Typography>
            </Paper>
          )}

          {data.length > 0 && columns.length > 0 && (
            <Paper sx={{ width: '100%', overflow: 'hidden' }}>
              <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h6">Data Explorer</Typography>
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

              {/* Formula Results Display */}
              {Object.entries(formulaResults).length > 0 && (
                <Box sx={{ p: 2, bgcolor: 'grey.100', borderBottom: 1, borderColor: 'divider' }}>
                  <Typography variant="subtitle2" gutterBottom>Formula Results:</Typography>
                  <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                    {Object.entries(formulaResults).map(([key, value]) => {
                      const [column, operation] = key.split('_');
                      return (
                        <Chip
                          key={key}
                          label={`${column} ${operation}: ${value.toFixed(2)}`}
                          color="primary"
                          variant="outlined"
                        />
                      );
                    })}
                  </Box>
                </Box>
              )}

              <TableContainer sx={{ maxHeight: 440, overflowX: 'auto' }}>
                <Table stickyHeader>
                  <TableHead>
                    <TableRow>
                      {columns.map((column) => (
                        <TableCell 
                          key={column}
                          sx={{ 
                            cursor: 'pointer',
                            '&:hover': { bgcolor: 'action.hover' }
                          }}
                          onClick={(e) => handleColumnMenuClick(e, column)}
                        >
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {column}
                            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                              <IconButton
                                size="small"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleSort(column);
                                }}
                              >
                                <SortIcon fontSize="small" />
                              </IconButton>
                              {sortConfig?.column === column && (
                                <Typography variant="caption" color="primary">
                                  {sortConfig.direction === 'asc' ? '↑' : '↓'}
                                </Typography>
                              )}
                            </Box>
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {getSortedData()
                      .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                      .map((row, rowIndex) => (
                        <TableRow
                          key={`row-${rowIndex}`}
                          sx={{
                            '&:nth-of-type(odd)': { bgcolor: 'grey.50' },
                            '&:hover': { bgcolor: 'grey.100' }
                          }}
                        >
                          {columns.map((column, colIndex) => (
                            <TableCell
                              key={`cell-${rowIndex}-${colIndex}`}
                              sx={{
                                fontSize: '0.875rem',
                                whiteSpace: 'nowrap',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                minWidth: 150
                              }}
                            >
                              {(() => {
                                const value = Array.isArray(row) ? row[colIndex] : row[column];
                                if (value === null || value === undefined || value === '') {
                                  return '-';
                                }
                                if (typeof value === 'object') {
                                  return JSON.stringify(value);
                                }
                                return String(value);
                              })()}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </TableContainer>
              <TablePagination
                rowsPerPageOptions={[10, 25, 50, 100]}
                component="div"
                count={data.length}
                rowsPerPage={rowsPerPage}
                page={page}
                onPageChange={handleChangePage}
                onRowsPerPageChange={handleChangeRowsPerPage}
              />
            </Paper>
          )}
        </Box>
      </TabPanel>

      <TabPanel value={activeTab} index={1}>
        <Box sx={{ p: 2 }}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">
                      Schema for {selectedTables.length > 0 ? selectedTables[0].name : 'No table selected'}
                    </Typography>
                    <Button
                      size="small"
                      onClick={handleSchemaTabClick}
                      startIcon={<FilterIcon />}
                      variant={isSchemaTabActive ? "contained" : "outlined"}
                      disabled={selectedTables.length === 0}
                    >
                      {isSchemaTabActive ? "Hide Schema" : "View Schema"}
                    </Button>
                  </Box>
                  {showSchemaAlert && schema.length > 0 && (
                    <Alert 
                      severity="info" 
                      onClose={() => setShowSchemaAlert(false)}
                      sx={{ mb: 2 }}
                    >
                      Found {schema.length} columns in the schema.
                    </Alert>
                  )}
                  {isSchemaTabActive && schema.length > 0 ? (
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell><strong>Column Name</strong></TableCell>
                            <TableCell><strong>Type</strong></TableCell>
                            <TableCell><strong>Description</strong></TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {schema.map((col) => (
                            <TableRow key={col.name}>
                              <TableCell>{col.name}</TableCell>
                              <TableCell>{col.type}</TableCell>
                              <TableCell>{col.comment}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  ) : schema.length > 0 ? (
                    <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', gap: 1 }}>
                      {schema.map((col) => (
                        <Chip 
                          key={col.name} 
                          label={`${col.name}: ${col.type}`} 
                          size="small"
                          title={col.comment}
                        />
                      ))}
                    </Stack>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      {selectedTables.length === 0 ? 'Please select a table to view its schema.' : 'No schema information available for this table.'}
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
      </TabPanel>

      <TabPanel value={activeTab} index={2}>
        {selectedTables.length > 0 ? (
          <TransformationTab
            tableName={selectedTables[0].name}
            userId="test_user"
          />
        ) : (
          <Box sx={{ p: 2 }}>
            <Typography variant="body1" color="text.secondary">
              Please select a table to view transformation options.
            </Typography>
          </Box>
        )}
      </TabPanel>

      <TabPanel value={activeTab} index={3}>
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Upload File
          </Typography>
          <FileUpload
            onUploadSuccess={handleUploadSuccess}
            onError={handleUploadError}
            selectedTable={selectedTables.length > 0 ? selectedTables[0] : undefined}
          />
        </Box>
      </TabPanel>

      <TabPanel value={activeTab} index={4}>
        <Box sx={{ p: 2 }}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Saved Queries</Typography>
                  <Button
                    variant="contained"
                    startIcon={<SaveIcon />}
                    onClick={() => setSaveQueryDialogOpen(true)}
                  >
                    Create Query
                  </Button>
                </Box>

                {loadingReports && (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                    <CircularProgress />
                  </Box>
                )}

                {reportError && (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    {reportError}
                  </Alert>
                )}

                {!loadingReports && savedQueries.length === 0 && (
                  <Alert severity="info">
                    No queries saved yet. Create a new query to get started.
                  </Alert>
                )}

                {!loadingReports && savedQueries.length > 0 && (
                  <List>
                    {savedQueries.map((savedQuery) => (
                      <ListItem
                        key={savedQuery.query_id}
                        sx={{
                          border: '1px solid',
                          borderColor: 'divider',
                          borderRadius: 1,
                          mb: 1,
                          '&:hover': {
                            bgcolor: 'action.hover',
                          },
                        }}
                      >
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="subtitle1">{savedQuery.name}</Typography>
                              {savedQuery.is_favorite && <StarIcon color="warning" />}
                            </Box>
                          }
                          secondary={
                            <>
                              <Typography variant="body2" color="text.secondary">
                                {savedQuery.description}
                              </Typography>
                              <Box sx={{ mt: 1 }}>
                                <Chip
                                  size="small"
                                  label={`Executed ${savedQuery.execution_count} times`}
                                  sx={{ mr: 1 }}
                                />
                                {savedQuery.last_run && (
                                  <Chip
                                    size="small"
                                    label={`Last run: ${new Date(savedQuery.last_run).toLocaleDateString()}`}
                                  />
                                )}
                              </Box>
                            </>
                          }
                        />
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Tooltip title="Execute">
                            <IconButton
                              onClick={() => handleExecuteQuery(savedQuery)}
                              color="primary"
                              disabled={loading}
                            >
                              <PlayArrowIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title={savedQuery.is_favorite ? "Remove from favorites" : "Add to favorites"}>
                            <IconButton
                              onClick={() => handleToggleFavoriteQuery(savedQuery.query_id, savedQuery.is_favorite)}
                              color={savedQuery.is_favorite ? 'warning' : 'default'}
                            >
                              {savedQuery.is_favorite ? <StarIcon /> : <StarBorderIcon />}
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton
                              onClick={() => handleDeleteQuery(savedQuery.query_id)}
                              color="error"
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </ListItem>
                    ))}
                  </List>
                )}
              </Paper>
            </Grid>
          </Grid>
        </Box>
      </TabPanel>

      <Dialog
        open={reportDialogOpen}
        onClose={handleCloseReportDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {reportDialogMode === 'edit' ? 'Edit Query' : 'Create New Query'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Query Name"
                value={reportForm.name}
                onChange={(e) => setReportForm(prev => ({ ...prev, name: e.target.value }))}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                value={reportForm.description}
                onChange={(e) => setReportForm(prev => ({ ...prev, description: e.target.value }))}
                multiline
                rows={2}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Query"
                value={reportForm.query}
                onChange={(e) => setReportForm(prev => ({ ...prev, query: e.target.value }))}
                multiline
                rows={4}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Schedule (Cron Expression)"
                value={reportForm.schedule}
                onChange={(e) => setReportForm(prev => ({ ...prev, schedule: e.target.value }))}
                placeholder="0 0 * * * (Daily at midnight)"
                helperText="Optional: Use cron expression to schedule query execution"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseReportDialog}>Cancel</Button>
          <Button
            onClick={handleSaveReport}
            variant="contained"
            disabled={!reportForm.name || !reportForm.query}
          >
            {reportDialogMode === 'edit' ? 'Update Query' : 'Save Query'}
          </Button>
        </DialogActions>
      </Dialog>

      <Menu
        anchorEl={formulaMenuAnchor}
        open={Boolean(formulaMenuAnchor)}
        onClose={handleFormulaClose}
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

      <Menu
        anchorEl={columnMenuAnchor}
        open={Boolean(columnMenuAnchor)}
        onClose={handleColumnMenuClose}
      >
        <MenuItem onClick={() => handleColumnOperation('sum')}>Sum</MenuItem>
        <MenuItem onClick={() => handleColumnOperation('average')}>Average</MenuItem>
        <MenuItem onClick={() => handleColumnOperation('min')}>Min</MenuItem>
        <MenuItem onClick={() => handleColumnOperation('max')}>Max</MenuItem>
      </Menu>

      {data.length > 0 && columns.length > 0 && (
        <>
          <Paper sx={{ width: '100%', overflow: 'hidden', mb: 2 }}>
            {/* ... existing data grid code ... */}
          </Paper>

          {/* Charts Section */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Charts</Typography>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={handleAddChart}
                size="small"
              >
                Add Chart
              </Button>
            </Box>

            <Grid container spacing={2}>
              {charts.map((chart) => (
                <Grid item xs={12} md={6} key={chart.id}>
                  <Card>
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Typography variant="h6">{chart.title}</Typography>
                        <Box>
                          <IconButton
                            size="small"
                            onClick={() => handleEditChart(chart)}
                          >
                            <EditIcon />
                          </IconButton>
                          <IconButton
                            size="small"
                            onClick={() => handleDeleteChart(chart.id)}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Box>
                      </Box>
                      {renderChart(chart)}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </>
      )}

      {/* Chart Configuration Dialog */}
      <Dialog
        open={chartDialogOpen}
        onClose={() => setChartDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {editingChart?.id ? 'Edit Chart' : 'Create New Chart'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Chart Title"
                value={editingChart?.title || ''}
                onChange={(e) => setEditingChart(prev => prev ? { ...prev, title: e.target.value } : null)}
              />
            </Grid>
            <Grid item xs={12}>
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
                  <MenuItem value="scatter">Scatter Plot</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>X-Axis</InputLabel>
                <Select
                  value={editingChart?.xAxis || ''}
                  onChange={(e) => setEditingChart(prev => prev ? { ...prev, xAxis: e.target.value } : null)}
                  label="X-Axis"
                >
                  {columns.map((column) => (
                    <MenuItem key={column} value={column}>
                      {column}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Y-Axis</InputLabel>
                <Select
                  multiple
                  value={editingChart?.yAxis || []}
                  onChange={(e) => setEditingChart(prev => prev ? { ...prev, yAxis: e.target.value as string[] } : null)}
                  label="Y-Axis"
                >
                  {columns.map((column) => (
                    <MenuItem key={column} value={column}>
                      {column}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            {editingChart?.type !== 'pie' && (
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>Aggregation</InputLabel>
                  <Select
                    value={editingChart?.aggregation || 'sum'}
                    onChange={(e) => setEditingChart(prev => prev ? { ...prev, aggregation: e.target.value as ChartConfig['aggregation'] } : null)}
                    label="Aggregation"
                  >
                    <MenuItem value="sum">Sum</MenuItem>
                    <MenuItem value="average">Average</MenuItem>
                    <MenuItem value="count">Count</MenuItem>
                    <MenuItem value="min">Minimum</MenuItem>
                    <MenuItem value="max">Maximum</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            )}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setChartDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleSaveChart}
            variant="contained"
            startIcon={<SaveIcon />}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog
        open={saveQueryDialogOpen}
        onClose={() => setSaveQueryDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Save SQL Query</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Query Name"
                value={saveQueryForm.name}
                onChange={(e) => setSaveQueryForm(prev => ({ ...prev, name: e.target.value }))}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                value={saveQueryForm.description}
                onChange={(e) => setSaveQueryForm(prev => ({ ...prev, description: e.target.value }))}
                multiline
                rows={2}
              />
            </Grid>
            <Grid item xs={12}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                SQL Query:
              </Typography>
              <Paper
                sx={{
                  p: 1,
                  bgcolor: 'grey.100',
                  borderRadius: 1,
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                }}
              >
                {query}
              </Paper>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveQueryDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleSaveQuery}
            variant="contained"
            disabled={!saveQueryForm.name || !query.trim()}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError(null)}
      >
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>

      <Snackbar
        open={!!success}
        autoHideDuration={6000}
        onClose={() => setSuccess(null)}
      >
        <Alert severity="success" onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DataExplorer; 