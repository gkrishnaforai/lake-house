import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  IconButton,
  Tooltip,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Stack,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Visibility as VisibilityIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { API_ENDPOINTS } from '../config';
import { CatalogInterfaceProps, FileMetadata } from '../types';

interface TableInfo {
  name: string;
  schema: any;
  location: string;
  description?: string;
  created_at: string;
  updated_at: string;
}

interface FileInfo {
  name: string;
  size: number;
  last_modified: string;
  format: string;
  location: string;
}

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const CatalogInterface: React.FC<CatalogInterfaceProps> = ({ onTableSelect, onFileSelect }) => {
  const [selectedTable, setSelectedTable] = useState<TableInfo | null>(null);
  const [selectedFile, setSelectedFile] = useState<FileInfo | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewData, setPreviewData] = useState<any[]>([]);
  const queryClient = useQueryClient();

  const { data: catalog, isLoading, error } = useQuery({
    queryKey: ['catalog'],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE_URL}/api/catalog`);
      return response.data;
    },
  });

  // Fetch tables from catalog
  const { data: tables, isLoading: isLoadingTables, error: tablesError } = useQuery<TableInfo[]>({
    queryKey: ['tables'],
    queryFn: async () => {
      const response = await axios.get(API_ENDPOINTS.CATALOG_TABLES);
      return response.data;
    }
  });

  // Fetch files from catalog
  const { data: files, isLoading: isLoadingFiles, error: filesError } = useQuery<FileInfo[]>({
    queryKey: ['files'],
    queryFn: async () => {
      const response = await axios.get(API_ENDPOINTS.CATALOG_FILES);
      return response.data;
    }
  });

  // Delete table mutation
  const deleteTableMutation = useMutation({
    mutationFn: async (tableName: string) => {
      await axios.delete(`${API_ENDPOINTS.CATALOG_TABLES}/${tableName}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tables'] });
    }
  });

  // Delete file mutation
  const deleteFileMutation = useMutation({
    mutationFn: async (filePath: string) => {
      await axios.delete(API_ENDPOINTS.FILE_DELETE(filePath));
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] });
    }
  });

  const handlePreviewFile = async (file: FileInfo) => {
    try {
      const response = await axios.get(API_ENDPOINTS.FILE_PREVIEW(file.location));
      setPreviewData(response.data.rows);
      setSelectedFile(file);
      setPreviewOpen(true);
    } catch (error) {
      console.error('Error previewing file:', error);
    }
  };

  const handleDeleteTable = async (table: TableInfo) => {
    if (window.confirm(`Are you sure you want to delete table "${table.name}"?`)) {
      deleteTableMutation.mutate(table.name);
    }
  };

  const handleDeleteFile = async (file: FileInfo) => {
    if (window.confirm(`Are you sure you want to delete file "${file.name}"?`)) {
      deleteFileMutation.mutate(file.location);
    }
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error instanceof Error) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="error">Error loading catalog: {error.message}</Alert>
      </Box>
    );
  }

  if (!catalog || !catalog.tables || catalog.tables.length === 0) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="info">No tables available in the catalog</Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Catalog Tables
      </Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Table Name</TableCell>
              <TableCell>Format</TableCell>
              <TableCell>Size</TableCell>
              <TableCell>Last Modified</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {catalog.tables.map((table: any) => (
              <TableRow key={table.name}>
                <TableCell>{table.name}</TableCell>
                <TableCell>{table.format}</TableCell>
                <TableCell>{(table.size / 1024).toFixed(2)} KB</TableCell>
                <TableCell>{new Date(table.last_modified).toLocaleString()}</TableCell>
                <TableCell>
                  <Button
                    size="small"
                    onClick={() => {
                      onTableSelect?.(table);
                      onFileSelect?.(null as unknown as FileMetadata);
                    }}
                  >
                    View Details
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default CatalogInterface; 