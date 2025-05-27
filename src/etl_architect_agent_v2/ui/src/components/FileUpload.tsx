import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Button,
  Typography,
  LinearProgress,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Alert,
  TextField,
  FormControlLabel,
  Switch,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TableContainer,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Description as FileIcon
} from '@mui/icons-material';
import { CatalogService, getUserId } from '../services/catalogService';
import { FileInfo, TableInfo } from '../types/api';

interface FileUploadProps {
  onUploadSuccess: () => void;
  onError: (error: string) => void;
  selectedTable?: TableInfo;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUploadSuccess, onError, selectedTable }) => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [tableName, setTableName] = useState('');
  const [createNew, setCreateNew] = useState(true);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewData, setPreviewData] = useState<any[] | null>(null);
  const [previewColumns, setPreviewColumns] = useState<string[]>([]);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [tableExists, setTableExists] = useState(false);
  const catalogService = new CatalogService();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Update tableName when selectedTable changes
  useEffect(() => {
    if (selectedTable) {
      setTableName(selectedTable.name);
      setCreateNew(false);
      setTableExists(true);
    }
  }, [selectedTable]);

  // Check if table exists when tableName changes
  useEffect(() => {
    const checkTableExists = async () => {
      if (!tableName || createNew) {
        setTableExists(false);
        return;
      }

      try {
        const userId = getUserId();
        const tables = await catalogService.listTables(userId);
        const exists = tables.some(table => table.name === tableName);
        setTableExists(exists);
        if (!exists) {
          setUploadError(`Table "${tableName}" does not exist. Please create a new table or select an existing one.`);
        } else {
          setUploadError(null);
        }
      } catch (error) {
        console.error('Error checking table existence:', error);
        setTableExists(false);
      }
    };

    checkTableExists();
  }, [tableName, createNew]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    console.log('File select triggered:', event.target.files);
    const file = event.target.files?.[0];
    if (file) {
      console.log('File selected:', file.name, file.type, file.size);
      setSelectedFile(file);
      setUploadError(null);
    }
  };

  const handleButtonClick = () => {
    console.log('Upload button clicked');
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || !tableName) {
      setUploadError('Please select a file and provide a table name');
      return;
    }

    if (!createNew && !tableExists) {
      setUploadError(`Table "${tableName}" does not exist. Please create a new table or select an existing one.`);
      return;
    }

    setUploading(true);
    setProgress(0);
    setUploadError(null);

    try {
      // Simulate upload progress
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(interval);
            return prev;
          }
          return prev + 10;
        });
      }, 500);

      const userId = getUserId();
      const result = await catalogService.uploadFile(selectedFile, tableName, userId, createNew);
      
      clearInterval(interval);
      setProgress(100);
      onUploadSuccess();
      setSelectedFile(null);
      if (!selectedTable) {
        setTableName('');
        setCreateNew(true);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      setUploadError(errorMessage);
      onError(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  const handlePreview = async (s3Path: string) => {
    setPreviewLoading(true);
    setPreviewError(null);
    setPreviewData(null);
    setPreviewColumns([]);
    setPreviewOpen(true);
    try {
      const userId = getUserId();
      const data = await catalogService.getFilePreview(s3Path, userId, 10);
      setPreviewData(data);
      setPreviewColumns(data && data[0] ? Object.keys(data[0]) : []);
    } catch (e: any) {
      setPreviewError(e?.message || 'Failed to load preview');
    } finally {
      setPreviewLoading(false);
    }
  };

  return (
    <Box sx={{ mt: 2 }}>
      <Paper
        sx={{
          p: 3,
          border: '2px dashed',
          borderColor: 'primary.main',
          borderRadius: 2,
          textAlign: 'center',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.json,.xlsx,.xls,.parquet"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          id="file-upload"
        />
        <Button
          variant="contained"
          onClick={handleButtonClick}
          startIcon={<UploadIcon />}
          disabled={uploading}
        >
          Select File
        </Button>

        {uploadError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {uploadError}
          </Alert>
        )}

        {selectedFile && (
          <Box sx={{ mt: 2 }}>
            <List>
              <ListItem>
                <ListItemIcon>
                  <FileIcon />
                </ListItemIcon>
                <ListItemText
                  primary={selectedFile.name}
                  secondary={`${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`}
                />
                <Chip
                  label={selectedFile.type || 'Unknown type'}
                  color="primary"
                  size="small"
                />
              </ListItem>
            </List>

            <Grid container spacing={2} sx={{ mt: 2 }}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Table Name"
                  value={tableName}
                  onChange={(e) => setTableName(e.target.value)}
                  disabled={uploading || !!selectedTable}
                  helperText={
                    selectedTable 
                      ? "Table name is set from selected table" 
                      : createNew 
                        ? "Enter a name for the new table" 
                        : tableExists 
                          ? "Table exists and will be updated" 
                          : "Table does not exist. Please create a new table or select an existing one."
                  }
                  error={!createNew && !tableExists && !!tableName}
                  required
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={createNew}
                      onChange={(e) => setCreateNew(e.target.checked)}
                      disabled={uploading || !!selectedTable}
                    />
                  }
                  label="Create New Table"
                />
                <Typography variant="caption" color="text.secondary" display="block">
                  {createNew
                    ? 'A new table will be created with this data'
                    : 'Data will be added to an existing table'}
                </Typography>
              </Grid>
            </Grid>

            {uploading && (
              <Box sx={{ mt: 2 }}>
                <LinearProgress variant="determinate" value={progress} />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Uploading... {progress}%
                </Typography>
              </Box>
            )}

            <Button
              variant="contained"
              onClick={handleUpload}
              disabled={uploading || !tableName}
              sx={{ mt: 2 }}
            >
              {uploading ? 'Uploading...' : 'Upload File'}
            </Button>
          </Box>
        )}

        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          Supported formats: CSV, JSON, Excel, Parquet
        </Typography>
      </Paper>
      <Dialog open={previewOpen} onClose={() => setPreviewOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>File Preview</DialogTitle>
        <DialogContent>
          {previewLoading && <Typography>Loading preview...</Typography>}
          {previewError && <Alert severity="error">{previewError}</Alert>}
          {previewData && previewData.length > 0 && (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {previewColumns.map((col) => (
                      <TableCell key={col}>{col}</TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {previewData.map((row, idx) => (
                    <TableRow key={idx}>
                      {previewColumns.map((col) => (
                        <TableCell key={col}>{row[col]}</TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
          {previewData && previewData.length === 0 && <Typography>No data to preview.</Typography>}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default FileUpload; 