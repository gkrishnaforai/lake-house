import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Button,
  LinearProgress,
  Alert,
  Grid,
  Card,
  CardContent,
  Chip,
  Stack,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  FormHelperText,
  CircularProgress,
  SelectChangeEvent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Explore as ExploreIcon,
  Save as SaveIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';
import { transformationService } from '../services/transformationService';
import { TransformationTool, TransformationResult } from '../types/transformation';

interface TransformationTabProps {
  tableName: string;
  userId: string;
}

interface TableColumn {
  name: string;
  type: string;
  description: string;
}

export const TransformationTab: React.FC<TransformationTabProps> = ({
  tableName,
  userId,
}) => {
  const [tools, setTools] = useState<TransformationTool[]>([]);
  const [selectedTool, setSelectedTool] = useState<string>('');
  const [sourceColumns, setSourceColumns] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [availableColumns, setAvailableColumns] = useState<TableColumn[]>([]);
  const [loadingColumns, setLoadingColumns] = useState(false);

  useEffect(() => {
    loadTools();
    loadTableColumns();
  }, [tableName]);

  const loadTools = async () => {
    try {
      const availableTools = await transformationService.getAvailableTransformations();
      setTools(availableTools);
    } catch (err) {
      setError('Failed to load transformation tools');
    }
  };

  const loadTableColumns = async () => {
    try {
      setLoadingColumns(true);
      const columns = await transformationService.getTableColumns(tableName, userId);
      setAvailableColumns(columns);
    } catch (err) {
      setError('Failed to load table columns');
    } finally {
      setLoadingColumns(false);
    }
  };

  const handleToolSelect = (event: SelectChangeEvent<string>) => {
    setSelectedTool(event.target.value);
    setError(null);
    setSuccess(null);
    setPreviewData([]);
  };

  const handleSourceColumnsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSourceColumns(event.target.value);
    setError(null);
    setSuccess(null);
    setPreviewData([]);
  };

  const handleApplyTransformation = async () => {
    if (!selectedTool || !sourceColumns) {
      setError('Please select a transformation tool and source columns');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch('/api/transformation/apply', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          table_name: tableName,
          tool_id: selectedTool,
          source_columns: sourceColumns.split(',').map(col => col.trim()),
          user_id: userId
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        let errorMessage = data.detail || 'An error occurred during transformation';
        
        if (errorMessage.includes('empty (0 bytes)')) {
          errorMessage = 'The selected table file is empty. Please ensure the file contains data before proceeding.';
        } else if (errorMessage.includes('does not exist in S3')) {
          errorMessage = 'The selected table file could not be found. Please ensure the file has been properly uploaded.';
        } else if (errorMessage.includes('No valid data rows found')) {
          errorMessage = 'The selected table contains no valid data. Please ensure the table has valid data before proceeding.';
        } else if (errorMessage.includes('No data rows found')) {
          errorMessage = 'The selected table contains no data. Please ensure the table has been properly populated.';
        }

        setError(errorMessage);
        return;
      }

      setSuccess('Transformation applied successfully');
      setPreviewData(data.preview_data);
    } catch (err) {
      setError('Failed to apply transformation. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const selectedToolConfig = tools.find(tool => tool.id === selectedTool);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Transform Data
      </Typography>

      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Transformation Tool</InputLabel>
        <Select
          value={selectedTool}
          label="Transformation Tool"
          onChange={handleToolSelect}
        >
          {tools.map((tool) => (
            <MenuItem key={tool.id} value={tool.id}>
              {tool.name}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {selectedToolConfig && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            {selectedToolConfig.description}
          </Typography>
          <Typography variant="subtitle2" gutterBottom>
            Example Input:
          </Typography>
          <Typography variant="body2" sx={{ mb: 1 }}>
            {selectedToolConfig.example_input}
          </Typography>
          <Typography variant="subtitle2" gutterBottom>
            Example Output:
          </Typography>
          <Typography variant="body2" sx={{ mb: 2 }}>
            {selectedToolConfig.example_output}
          </Typography>
        </Box>
      )}

      <FormControl fullWidth sx={{ mb: 2 }}>
        <TextField
          label="Source Columns"
          value={sourceColumns}
          onChange={handleSourceColumnsChange}
          helperText="Enter column names separated by commas"
        />
      </FormControl>

      {loadingColumns ? (
        <CircularProgress size={24} />
      ) : (
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Available Columns:
          </Typography>
          <TableContainer component={Paper}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Description</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {availableColumns.map((column) => (
                  <TableRow key={column.name}>
                    <TableCell>{column.name}</TableCell>
                    <TableCell>{column.type}</TableCell>
                    <TableCell>{column.description}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}

      <Button
        variant="contained"
        onClick={handleApplyTransformation}
        disabled={loading || !selectedTool || !sourceColumns}
        sx={{ mb: 2 }}
      >
        {loading ? <CircularProgress size={24} /> : 'Apply Transformation'}
      </Button>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      {previewData.length > 0 && (
        <Box>
          <Typography variant="subtitle1" gutterBottom>
            Preview
          </Typography>
          <TableContainer component={Paper}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  {Object.keys(previewData[0]).map((key) => (
                    <TableCell key={key}>{key}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {previewData.map((row, index) => (
                  <TableRow key={index}>
                    {Object.values(row).map((value, i) => (
                      <TableCell key={i}>
                        {String(value)}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}
    </Box>
  );
}; 