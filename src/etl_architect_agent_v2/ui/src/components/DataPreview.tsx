import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

interface DataPreviewProps {
  filePath: string;
}

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const DataPreview: React.FC<DataPreviewProps> = ({ filePath }) => {
  const { data: previewData, isLoading, error } = useQuery({
    queryKey: ['preview', filePath],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE_URL}/api/catalog/files/${filePath}/preview`);
      return response.data;
    },
    enabled: !!filePath,
  });

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
        <Alert severity="error">Error loading preview: {error.message}</Alert>
      </Box>
    );
  }

  if (!previewData || !previewData.data || previewData.data.length === 0) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="warning">No preview data available</Alert>
      </Box>
    );
  }

  const columns = Object.keys(previewData.data[0]);

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Data Preview
      </Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              {columns.map((column) => (
                <TableCell key={column}>{column}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {previewData.data.map((row: any, rowIndex: number) => (
              <TableRow key={rowIndex}>
                {columns.map((column) => (
                  <TableCell key={`${rowIndex}-${column}`}>
                    {row[column]?.toString() || '-'}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default DataPreview; 