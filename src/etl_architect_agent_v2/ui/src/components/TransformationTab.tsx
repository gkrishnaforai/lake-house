import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Backdrop,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Explore as ExploreIcon,
  Save as SaveIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  Remove as RemoveIcon,
  Warning as WarningIcon,
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
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [availableColumns, setAvailableColumns] = useState<TableColumn[]>([]);
  const [loadingColumns, setLoadingColumns] = useState(false);
  const [showCancelDialog, setShowCancelDialog] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isCancelling, setIsCancelling] = useState(false);

  useEffect(() => {
    loadTools();
    loadTableColumns();
  }, [tableName]);

  const loadTools = async () => {
    try {
      const availableTools = await transformationService.getAvailableTransformations();
      setTools(availableTools);
    } catch (err) {
      setError('Failed to load classification tools');
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

  const handleColumnSelect = (columnName: string) => {
    setSelectedColumns(prev => {
      if (prev.includes(columnName)) {
        return prev.filter(col => col !== columnName);
      }
      return [...prev, columnName];
    });
    setError(null);
    setSuccess(null);
    setPreviewData([]);
  };

  const handleApplyTransformation = async () => {
    if (!selectedTool) {
      setError('Please select a classification tool');
      return;
    }
    if (selectedColumns.length === 0) {
      setError('Please select at least one column');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);
    setProgress(0);

    // Simulate progress updates
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return prev;
        }
        return prev + 10;
      });
    }, 1000);

    try {
      const response = await fetch('/api/transformation/apply', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          table_name: tableName,
          tool_id: selectedTool,
          source_columns: selectedColumns,
          user_id: userId
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        let errorMessage = data.detail || 'An error occurred during classification';
        
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

      setProgress(100);
      setSuccess('Classification applied successfully');
      setPreviewData(data.preview_data);
    } catch (err) {
      setError('Failed to apply classification. Please try again.');
    } finally {
      clearInterval(progressInterval);
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setShowCancelDialog(true);
  };

  const handleConfirmCancel = () => {
    setIsCancelling(true);
    setLoading(false);
    setProgress(0);
    setShowCancelDialog(false);
    setError('Classification cancelled by user');
    // Add any cleanup logic here
    setTimeout(() => {
      setIsCancelling(false);
    }, 1000);
  };

  const selectedToolConfig = tools.find(tool => tool.id === selectedTool);

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">
          Data Classification & Categorization
        </Typography>
        <Button
          variant="contained"
          onClick={handleApplyTransformation}
          disabled={loading || !selectedTool || selectedColumns.length === 0}
          startIcon={loading ? <CircularProgress size={20} /> : <PlayIcon />}
        >
          Apply Classification
        </Button>
      </Box>

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

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Classification Tool
              </Typography>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Select Tool</InputLabel>
                <Select
                  value={selectedTool}
                  label="Select Tool"
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
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Source Columns
              </Typography>
              {loadingColumns ? (
                <CircularProgress size={24} />
              ) : (
                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Select columns to transform:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                    {selectedColumns.map((column) => (
                      <Chip
                        key={column}
                        label={column}
                        onDelete={() => handleColumnSelect(column)}
                        color="primary"
                      />
                    ))}
                  </Box>
                  <TableContainer 
                    component={Paper} 
                    sx={{ 
                      maxHeight: 300,
                      '&::-webkit-scrollbar': {
                        width: '8px',
                        height: '8px',
                      },
                      '&::-webkit-scrollbar-track': {
                        background: '#f1f1f1',
                        borderRadius: '4px',
                      },
                      '&::-webkit-scrollbar-thumb': {
                        background: '#1976d2',
                        borderRadius: '4px',
                        '&:hover': {
                          background: '#1565c0',
                        },
                      },
                      '&::-webkit-scrollbar-corner': {
                        background: '#f1f1f1',
                      },
                    }}
                  >
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Name</TableCell>
                          <TableCell>Type</TableCell>
                          <TableCell>Description</TableCell>
                          <TableCell>Action</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {availableColumns.map((column) => (
                          <TableRow 
                            key={column.name}
                            hover
                            onClick={() => handleColumnSelect(column.name)}
                            sx={{ 
                              cursor: 'pointer',
                              backgroundColor: selectedColumns.includes(column.name) ? 'action.selected' : 'inherit'
                            }}
                          >
                            <TableCell>{column.name}</TableCell>
                            <TableCell>{column.type}</TableCell>
                            <TableCell>{column.description}</TableCell>
                            <TableCell>
                              <IconButton 
                                size="small"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleColumnSelect(column.name);
                                }}
                              >
                                {selectedColumns.includes(column.name) ? <RemoveIcon /> : <AddIcon />}
                              </IconButton>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {previewData.length > 0 && (
        <Box sx={{ mt: 3 }}>
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

      <Backdrop
        sx={{
          color: '#fff',
          zIndex: (theme) => theme.zIndex.drawer + 1,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
        }}
        open={loading}
      >
        <Box sx={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center',
          gap: 2,
          p: 4,
          borderRadius: 2,
          backgroundColor: 'background.paper',
          minWidth: 400,
        }}>
          <Typography variant="h6" gutterBottom>
            Applying Classification
          </Typography>
          <Box sx={{ width: '100%', mb: 2 }}>
            <LinearProgress 
              variant="determinate" 
              value={progress} 
              sx={{
                height: 10,
                borderRadius: 5,
                backgroundColor: 'grey.200',
                '& .MuiLinearProgress-bar': {
                  borderRadius: 5,
                  backgroundColor: 'primary.main',
                },
              }}
            />
            <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 1 }}>
              {progress}% Complete
            </Typography>
          </Box>
          <Button
            variant="outlined"
            color="error"
            onClick={handleCancel}
            disabled={isCancelling}
            startIcon={<WarningIcon />}
          >
            Cancel Classification
          </Button>
        </Box>
      </Backdrop>

      <Dialog
        open={showCancelDialog}
        onClose={() => setShowCancelDialog(false)}
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <WarningIcon color="warning" />
            Cancel Classification
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to cancel the classification? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowCancelDialog(false)}>
            Continue Classification
          </Button>
          <Button 
            onClick={handleConfirmCancel} 
            color="error" 
            variant="contained"
          >
            Cancel
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}; 