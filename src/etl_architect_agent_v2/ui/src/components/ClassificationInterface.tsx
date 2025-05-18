import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Grid,
  TextField,
  Typography,
  Alert,
  Paper,
  Stack,
  Chip,
  LinearProgress,
} from '@mui/material';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import {
  Assessment as AssessmentIcon,
  TableChart as TableChartIcon,
  DataObject as DataObjectIcon,
} from '@mui/icons-material';

interface ClassificationState {
  status: string;
  progress: number;
  current_operation: string;
  error?: string;
}

interface QualityMetrics {
  completeness: number;
  uniqueness: number;
  accuracy: number;
}

interface ColumnInfo {
  name: string;
  type: string;
  quality_metrics: QualityMetrics;
}

interface TableInfo {
  name: string;
  columns: ColumnInfo[];
  quality_score: number;
}

const QualityScore = ({ score }: { score: number }) => {
  const getColor = (score: number) => {
    if (score >= 90) return 'success';
    if (score >= 70) return 'warning';
    return 'error';
  };

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <LinearProgress
        variant="determinate"
        value={score}
        color={getColor(score) as any}
        sx={{ width: 100, height: 8, borderRadius: 4 }}
      />
      <Typography variant="body2" color="text.secondary">
        {score}%
      </Typography>
    </Box>
  );
};

const ClassificationInterface = () => {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [instruction, setInstruction] = useState('');
  const [state, setState] = useState<ClassificationState>({
    status: 'idle',
    progress: 0,
    current_operation: '',
  });
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<any>(null);
  const [clientId] = useState('default');
  const [tableInfo, setTableInfo] = useState<TableInfo | null>(null);

  useEffect(() => {
    if (selectedFile) {
      fetchTableInfo(selectedFile);
    }
  }, [selectedFile]);

  const fetchTableInfo = async (fileName: string) => {
    try {
      const response = await fetch(`/api/catalog/tables/${fileName}`);
      if (!response.ok) {
        throw new Error('Failed to fetch table info');
      }
      const data = await response.json();
      setTableInfo(data);
    } catch (err) {
      console.error('Error fetching table info:', err);
    }
  };

  const startClassification = useCallback(async () => {
    if (!selectedFile || !instruction) {
      setError('Please select a file and provide classification instructions');
      return;
    }

    try {
      const response = await fetch('/api/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          client_id: clientId,
          data_path: selectedFile,
          user_instruction: instruction,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start classification');
      }

      setState({
        status: 'pending',
        progress: 0,
        current_operation: 'Starting classification...',
      });
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    }
  }, [clientId, selectedFile, instruction]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file.name);
    }
  };

  const columns: GridColDef[] = preview?.sample_data?.[0] 
    ? Object.keys(preview.sample_data[0]).map((key) => ({
        field: key,
        headerName: key,
        flex: 1,
      }))
    : [];

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Data Classification
      </Typography>

      <Grid container spacing={3}>
        {/* Input Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Classification Settings
              </Typography>

              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv,.xlsx,.xls"
                  style={{ display: 'none' }}
                  id="file-input"
                  type="file"
                  onChange={handleFileSelect}
                />
                <label htmlFor="file-input">
                  <Button variant="contained" component="span">
                    Select File
                  </Button>
                </label>
                {selectedFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Selected: {selectedFile}
                  </Typography>
                )}
              </Box>

              <TextField
                fullWidth
                multiline
                rows={4}
                label="Classification Instructions"
                value={instruction}
                onChange={(e) => setInstruction(e.target.value)}
                sx={{ mb: 2 }}
              />

              <Button
                variant="contained"
                color="primary"
                onClick={startClassification}
                disabled={state.status === 'in_progress'}
              >
                Start Classification
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Status Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Classification Status
              </Typography>

              {state.status === 'in_progress' && (
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <CircularProgress
                    variant="determinate"
                    value={state.progress}
                    sx={{ mr: 2 }}
                  />
                  <Typography>
                    {state.current_operation} ({Math.round(state.progress)}%)
                  </Typography>
                </Box>
              )}

              {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {error}
                </Alert>
              )}

              {tableInfo && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Table Quality Metrics
                  </Typography>
                  <Paper sx={{ p: 2, mb: 2 }}>
                    <Stack spacing={2}>
                      <Box>
                        <Typography variant="subtitle2" gutterBottom>
                          Overall Quality Score
                        </Typography>
                        <QualityScore score={tableInfo.quality_score} />
                      </Box>
                      <Box>
                        <Typography variant="subtitle2" gutterBottom>
                          Column Quality
                        </Typography>
                        <Stack spacing={1}>
                          {tableInfo.columns.map((column) => (
                            <Box key={column.name}>
                              <Typography variant="body2" color="text.secondary">
                                {column.name}
                              </Typography>
                              <Grid container spacing={2}>
                                <Grid item xs={4}>
                                  <Typography variant="caption" color="text.secondary">
                                    Completeness
                                  </Typography>
                                  <QualityScore score={column.quality_metrics.completeness} />
                                </Grid>
                                <Grid item xs={4}>
                                  <Typography variant="caption" color="text.secondary">
                                    Uniqueness
                                  </Typography>
                                  <QualityScore score={column.quality_metrics.uniqueness} />
                                </Grid>
                                <Grid item xs={4}>
                                  <Typography variant="caption" color="text.secondary">
                                    Accuracy
                                  </Typography>
                                  <QualityScore score={column.quality_metrics.accuracy} />
                                </Grid>
                              </Grid>
                            </Box>
                          ))}
                        </Stack>
                      </Box>
                    </Stack>
                  </Paper>
                </Box>
              )}

              {preview && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Data Preview
                  </Typography>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Sample data from the selected file
                    </Typography>
                    <DataGrid
                      rows={preview.sample_data.map((row: any, index: number) => ({
                        id: index,
                        ...row,
                      }))}
                      columns={columns}
                      autoHeight
                      initialState={{
                        pagination: {
                          paginationModel: {
                            pageSize: 5,
                          },
                        },
                      }}
                      pageSizeOptions={[5]}
                    />
                  </Paper>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ClassificationInterface; 