import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Grid,
  Divider,
  CircularProgress,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import { Visibility as PreviewIcon } from '@mui/icons-material';
import { TableInfo } from '../types/api';
import { CatalogService } from '../services/catalogService';
import { useQuery } from '@tanstack/react-query';

interface TableDetailsDialogProps {
  open: boolean;
  onClose: () => void;
  table: TableInfo | null;
}

const TableDetailsDialog: React.FC<TableDetailsDialogProps> = ({ open, onClose, table }) => {
  const catalogService = new CatalogService();
  const [previewFile, setPreviewFile] = useState<string | null>(null);
  const [previewData, setPreviewData] = useState<any[] | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  const { data: schema, isLoading: isLoadingSchema, error: schemaError } = useQuery({
    queryKey: ['tableSchema', table?.name],
    queryFn: async () => {
      if (!table) return null;
      return await catalogService.getTableSchema(table.name);
    },
    enabled: !!table
  });

  const { data: files, isLoading: isLoadingFiles, error: filesError } = useQuery({
    queryKey: ['tableFiles', table?.name],
    queryFn: async () => {
      if (!table) return [];
      return await catalogService.getTableFiles(table.name);
    },
    enabled: !!table
  });

  const formatDate = (dateString: string | undefined): string => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleDateString();
    } catch (e) {
      return 'Invalid Date';
    }
  };

  const handlePreview = async (s3Path: string) => {
    setPreviewFile(s3Path);
    setPreviewLoading(true);
    setPreviewError(null);
    setPreviewData(null);
    try {
      // Extract bucket and key from s3Path
      const pathParts = s3Path.replace('s3://', '').split('/');
      const bucket = pathParts[0];
      const key = pathParts.slice(1).join('/');
      const data = await catalogService.getFilePreview(s3Path, 'test_user', 10);
      setPreviewData(data);
    } catch (e: any) {
      setPreviewError(e?.message || 'Failed to load preview');
    } finally {
      setPreviewLoading(false);
    }
  };

  if (!table) return null;

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        <Typography variant="h6">Table Details: {table.name}</Typography>
      </DialogTitle>
      <DialogContent dividers>
        <Grid container spacing={3}>
          {/* Basic Info */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>Description</Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              {table.description || 'No description available'}
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <Chip 
                label={`Created: ${formatDate(table.created_at)}`} 
                size="small" 
                variant="outlined" 
              />
              <Chip 
                label={`Updated: ${formatDate(table.updated_at)}`} 
                size="small" 
                variant="outlined" 
              />
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Divider />
          </Grid>

          {/* Schema */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>Schema</Typography>
            {isLoadingSchema ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                <CircularProgress size={24} />
              </Box>
            ) : schemaError ? (
              <Alert severity="error">Error loading schema</Alert>
            ) : schema?.schema ? (
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Column Name</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Description</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {schema.schema.map((col: any) => (
                      <TableRow key={col.name}>
                        <TableCell>{col.name}</TableCell>
                        <TableCell>{col.type}</TableCell>
                        <TableCell>{col.comment || '-'}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No schema information available
              </Typography>
            )}
          </Grid>

          <Grid item xs={12}>
            <Divider />
          </Grid>

          {/* Files */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>Associated Files</Typography>
            {isLoadingFiles ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                <CircularProgress size={24} />
              </Box>
            ) : filesError ? (
              <Alert severity="error">Error loading files</Alert>
            ) : files && files.length > 0 ? (
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>File Name</TableCell>
                      <TableCell align="right">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {files.map((file: any) => (
                      <TableRow key={file.name}>
                        <TableCell>{file.name}</TableCell>
                        <TableCell align="right">
                          <Tooltip title="Preview File">
                            <IconButton
                              size="small"
                              onClick={() => handlePreview(file.location)}
                              disabled={previewLoading}
                            >
                              <PreviewIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No files associated with this table
              </Typography>
            )}
          </Grid>

          {/* Preview Dialog */}
          {previewFile && (
            <Grid item xs={12}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Preview: {previewFile.split('/').pop()}
                </Typography>
                {previewLoading ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                    <CircularProgress size={24} />
                  </Box>
                ) : previewError ? (
                  <Alert severity="error">{previewError}</Alert>
                ) : previewData && previewData.length > 0 ? (
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          {Object.keys(previewData[0]).map((col) => (
                            <TableCell key={col}>{col}</TableCell>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {previewData.map((row, idx) => (
                          <TableRow key={idx}>
                            {Object.keys(previewData[0]).map((col) => (
                              <TableCell key={col}>{row[col]}</TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No data to preview
                  </Typography>
                )}
              </Paper>
            </Grid>
          )}
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default TableDetailsDialog; 