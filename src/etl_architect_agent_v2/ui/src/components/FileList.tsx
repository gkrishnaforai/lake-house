import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Tooltip,
  Typography,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { FileInfo } from '../types/api';

interface FileListProps {
  files: FileInfo[];
  onDelete: (s3Path: string) => void;
  onPreview: (s3Path: string) => void;
}

export const FileList: React.FC<FileListProps> = ({ files, onDelete, onPreview }) => {
  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'success':
        return 'success';
      case 'error':
        return 'error';
      case 'processing':
        return 'warning';
      case 'unknown':
        return 'default';
      default:
        return 'default';
    }
  };

  const getStatusLabel = (status: string) => {
    if (!status) return 'Unknown';
    return status.charAt(0).toUpperCase() + status.slice(1).toLowerCase();
  };

  const getTableName = (file: FileInfo) => {
    if (file.table_name) return file.table_name;
    // Try to extract table name from s3_path if available
    if (file.s3_path) {
      const parts = file.s3_path.split('/');
      const tableIndex = parts.findIndex(part => part === 'tables');
      if (tableIndex !== -1 && parts[tableIndex + 1]) {
        return parts[tableIndex + 1];
      }
    }
    return '-';
  };

  return (
    <TableContainer component={Paper} sx={{ mt: 2 }}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Table Name</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>File Name</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {files.length === 0 ? (
            <TableRow>
              <TableCell colSpan={4} align="center">
                <Typography variant="body2" color="text.secondary">
                  No files uploaded yet
                </Typography>
              </TableCell>
            </TableRow>
          ) : (
            files.map((file) => (
              <TableRow key={file.s3_path}>
                <TableCell>
                  <Typography variant="body2" noWrap>
                    {file.file_name || file.s3_path?.split('/').pop() || 'Unknown'}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip
                    label={getStatusLabel(file.status)}
                    color={getStatusColor(file.status) as any}
                    size="small"
                  />
                  {file.error && (
                    <Tooltip title={file.error}>
                      <Chip
                        label="Error"
                        color="error"
                        size="small"
                        sx={{ ml: 1 }}
                      />
                    </Tooltip>
                  )}
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {getTableName(file)}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Tooltip title="Preview">
                    <IconButton
                      size="small"
                      onClick={() => onPreview(file.s3_path)}
                      disabled={file.status === 'error'}
                    >
                      <VisibilityIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Delete">
                    <IconButton
                      size="small"
                      onClick={() => onDelete(file.s3_path)}
                      sx={{ ml: 1 }}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
}; 