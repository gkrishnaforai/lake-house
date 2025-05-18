import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios, { AxiosError } from 'axios';
import { API_ENDPOINTS } from '../config';
import { Box, Typography, Link, CircularProgress, Alert } from '@mui/material';
import { FileExplorerProps } from '../types';

interface CatalogInfo {
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
}

interface TableInfo {
  name: string;
  schema: any;
  location: string;
  description: string;
  created_at: string;
  updated_at: string;
}

interface FileInfo {
  name: string;
  size: number;
  last_modified: string;
  format: string;
  location: string;
  schema: any;
  quality: any;
}

interface ErrorResponse {
  detail: string;
}

const FileExplorer: React.FC<FileExplorerProps> = ({ onFileSelect, onTableSelect }) => {
  const [selectedTable, setSelectedTable] = useState<string | null>(null);

  // Fetch catalog info
  const { data: catalogInfo, isLoading: isLoadingCatalog, error: catalogError } = useQuery<CatalogInfo, AxiosError<ErrorResponse>>({
    queryKey: ['catalog'],
    queryFn: async () => {
      const response = await axios.get<CatalogInfo>(API_ENDPOINTS.CATALOG);
      return response.data;
    }
  });

  // Fetch tables
  const { data: tables, isLoading: isLoadingTables, error: tablesError } = useQuery<TableInfo[], AxiosError<ErrorResponse>>({
    queryKey: ['tables'],
    queryFn: async () => {
      const response = await axios.get<TableInfo[]>(API_ENDPOINTS.CATALOG_TABLES);
      return response.data;
    }
  });

  // Fetch files for selected table
  const { data: files, isLoading: isLoadingFiles, error: filesError } = useQuery<FileInfo[], AxiosError<ErrorResponse>>({
    queryKey: ['files', selectedTable],
    queryFn: async () => {
      if (!selectedTable) return [];
      const response = await axios.get<FileInfo[]>(API_ENDPOINTS.CATALOG_FILES);
      return response.data.filter((file) => 
        file.location.includes(selectedTable)
      );
    },
    enabled: !!selectedTable
  });

  if (isLoadingCatalog || isLoadingTables) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (catalogError || tablesError) {
    return (
      <Alert severity="error">
        Error loading data: {catalogError?.response?.data?.detail || tablesError?.response?.data?.detail || 'Unknown error'}
      </Alert>
    );
  }

  return (
    <Box p={2}>
      {/* Catalog Section */}
      <Box mb={3}>
        <Typography variant="h5" gutterBottom>
          <Link href="#" underline="hover" color="primary">
            {catalogInfo?.name || 'Data Catalog'}
          </Link>
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {catalogInfo?.description || 'Main data catalog'}
        </Typography>
      </Box>

      {/* Tables Section */}
      <Box mb={3}>
        <Typography variant="h6" gutterBottom>
          Tables
        </Typography>
        {tables && tables.length > 0 ? (
          <Box component="ul" sx={{ listStyle: 'none', pl: 0 }}>
            {tables.map((table: TableInfo) => (
              <Box component="li" key={table.name} mb={1}>
                <Link
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    setSelectedTable(table.name);
                  }}
                  underline="hover"
                  color={selectedTable === table.name ? 'secondary' : 'primary'}
                >
                  {table.name}
                </Link>
              </Box>
            ))}
          </Box>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No tables found
          </Typography>
        )}
      </Box>

      {/* Files Section */}
      {selectedTable && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Files in {selectedTable}
          </Typography>
          {isLoadingFiles ? (
            <CircularProgress size={20} />
          ) : filesError ? (
            <Alert severity="error">
              Error loading files: {filesError.response?.data?.detail || 'Unknown error'}
            </Alert>
          ) : files && files.length > 0 ? (
            <Box component="ul" sx={{ listStyle: 'none', pl: 0 }}>
              {files.map((file: FileInfo) => (
                <Box component="li" key={file.name} mb={1}>
                  <Link
                    href="#"
                    onClick={(e) => {
                      e.preventDefault();
                      onFileSelect(file);
                      onTableSelect?.(null);
                    }}
                    underline="hover"
                    color="primary"
                  >
                    {file.name} ({file.format})
                  </Link>
                </Box>
              ))}
            </Box>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No files found
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );
};

export default FileExplorer; 