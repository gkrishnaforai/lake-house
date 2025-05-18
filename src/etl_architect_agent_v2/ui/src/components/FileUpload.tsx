import React, { useState } from 'react';
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
import { FileList } from './FileList';
import { FileInfo } from '../types/api';

interface FileUploadProps {
  onUploadSuccess: () => void;
  onError: (error: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUploadSuccess, onError }) => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [tableName, setTableName] = useState('');
  const [createNew, setCreateNew] = useState(false);
  const [userFiles, setUserFiles] = useState<FileInfo[]>([]);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewData, setPreviewData] = useState<any[] | null>(null);
  const [previewColumns, setPreviewColumns] = useState<string[]>([]);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [deleteLoading, setDeleteLoading] = useState<string | null>(null); // s3Path being deleted
  const catalogService = new CatalogService();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setUploadError(null);
    }
  };

  const fetchUserFiles = async () => {
    try {
      const files = await catalogService.listUserFiles(getUserId());
      // Ensure each file has the required properties
      const processedFiles = files.map(file => ({
        ...file,
        file_name: file.file_name || file.s3_path?.split('/').pop() || 'Unknown',
        table_name: file.table_name || file.s3_path?.split('/').find(part => part === 'tables') || '-',
        status: file.status || 'unknown'
      }));
      setUserFiles(processedFiles);
    } catch (e) {
      console.error('Error fetching user files:', e);
      onError('Failed to fetch uploaded files');
    }
  };

  React.useEffect(() => {
    fetchUserFiles();
  }, []);

  const handleUpload = async () => {
    if (!selectedFile || !tableName) {
      setUploadError('Please select a file and provide a table name');
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
      setTableName('');
      setCreateNew(false);
      
      // Update the file list with the new file
      const newFile: FileInfo = {
        file_name: selectedFile.name,
        s3_path: result.location || '', // Use location from TableInfo
        status: 'success',
        table_name: tableName,
        size: selectedFile.size,
        last_modified: new Date().toISOString()
      };
      
      setUserFiles(prev => [newFile, ...prev]);
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

  const handleDelete = async (s3Path: string) => {
    if (!window.confirm('Are you sure you want to delete this file?')) return;
    setDeleteLoading(s3Path);
    try {
      const userId = getUserId();
      await catalogService.deleteFile(s3Path, userId);
      fetchUserFiles();
    } catch (e: any) {
      onError(e?.message || 'Failed to delete file');
    } finally {
      setDeleteLoading(null);
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
          type="file"
          accept=".csv,.json,.xlsx,.xls,.parquet"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          id="file-upload"
        />
        <label htmlFor="file-upload">
          <Button
            variant="contained"
            component="span"
            startIcon={<UploadIcon />}
            disabled={uploading}
          >
            Select File
          </Button>
        </label>

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
                  disabled={uploading}
                  helperText="Enter a name for the table where the data will be stored"
                  required
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={createNew}
                      onChange={(e) => setCreateNew(e.target.checked)}
                      disabled={uploading}
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
      <Typography variant="h6" sx={{ mt: 4, mb: 1 }}>
        Your Uploaded Files
      </Typography>
      <FileList files={userFiles} onDelete={handleDelete} onPreview={handlePreview} />
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