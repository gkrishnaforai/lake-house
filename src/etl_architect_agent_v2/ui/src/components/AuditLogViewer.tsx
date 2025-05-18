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
import { AuditLogViewerProps } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const AuditLogViewer: React.FC<AuditLogViewerProps> = ({ selectedTable, selectedFile }) => {
  const { data: auditLogs, isLoading, error } = useQuery({
    queryKey: ['audit', selectedTable || selectedFile],
    queryFn: async () => {
      const target = selectedTable || selectedFile;
      if (!target) return null;
      const response = await axios.get(`${API_BASE_URL}/api/catalog/audit/${target}`);
      return response.data;
    },
    enabled: !!selectedTable || !!selectedFile,
  });

  if (!selectedTable && !selectedFile) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="info">Select a table or file to view its audit logs</Alert>
      </Box>
    );
  }

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
        <Alert severity="error">Error loading audit logs: {error.message}</Alert>
      </Box>
    );
  }

  if (!auditLogs || auditLogs.length === 0) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="warning">No audit logs available</Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Audit Logs
      </Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Timestamp</TableCell>
              <TableCell>Action</TableCell>
              <TableCell>User</TableCell>
              <TableCell>Details</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {auditLogs.map((log: any, index: number) => (
              <TableRow key={index}>
                <TableCell>{new Date(log.timestamp).toLocaleString()}</TableCell>
                <TableCell>{log.action}</TableCell>
                <TableCell>{log.user}</TableCell>
                <TableCell>{log.details}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default AuditLogViewer; 