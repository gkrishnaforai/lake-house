import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { QueryInterfaceProps } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

interface QueryResult {
  columns: string[];
  rows: any[];
  execution_time: number;
  row_count: number;
}

const QueryInterface: React.FC<QueryInterfaceProps> = ({ selectedTable, selectedFile }) => {
  const [query, setQuery] = useState('');
  const [isExecuting, setIsExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);

  // Fetch available tables from catalog
  const { data: tables, isLoading: isLoadingTables } = useQuery({
    queryKey: ['tables'],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE_URL}/api/catalog/tables`);
      return response.data;
    },
  });

  const { data: queryResult, refetch } = useQuery({
    queryKey: ['query', query],
    queryFn: async () => {
      if (!query) return null;
      const response = await axios.post(`${API_BASE_URL}/api/catalog/query`, { query });
      return response.data;
    },
    enabled: false,
  });

  const handleExecuteQuery = async () => {
    if (!query) {
      setError('Please enter a query');
      return;
    }

    setIsExecuting(true);
    setError(null);

    try {
      await refetch();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Query execution failed');
    } finally {
      setIsExecuting(false);
    }
  };

  const handleTableSelect = (tableName: string | null) => {
    if (tableName) {
      setQuery(`SELECT * FROM ${tableName} LIMIT 100`);
    }
  };

  return (
    <Box>
      <Stack spacing={3}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Query Builder
          </Typography>
          <Stack spacing={2}>
            <FormControl fullWidth>
              <InputLabel>Select Table</InputLabel>
              <Select
                value={selectedTable}
                label="Select Table"
                onChange={(e) => handleTableSelect(e.target.value)}
                disabled={isLoadingTables}
              >
                <MenuItem value="">
                  <em>Select a table</em>
                </MenuItem>
                {tables?.map((table: any) => (
                  <MenuItem key={table.name} value={table.name}>
                    {table.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              fullWidth
              multiline
              rows={4}
              label="SQL Query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={isExecuting}
              placeholder="Enter your SQL query here..."
            />

            <Button
              variant="contained"
              onClick={handleExecuteQuery}
              disabled={isExecuting || !query.trim()}
              startIcon={isExecuting ? <CircularProgress size={20} /> : null}
            >
              {isExecuting ? 'Executing...' : 'Execute Query'}
            </Button>
          </Stack>
        </Paper>

        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {isExecuting && (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        )}

        {queryResult && (
          <Paper sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h6">Query Results</Typography>
                <Typography variant="body2" color="text.secondary">
                  {queryResult.row_count} rows • {queryResult.execution_time.toFixed(2)}s
                </Typography>
              </Box>

              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      {Object.keys(queryResult[0] || {}).map((column) => (
                        <TableCell key={column}>{column}</TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {queryResult.map((row: any, index: number) => (
                      <TableRow key={index}>
                        {Object.values(row).map((value: any, colIndex: number) => (
                          <TableCell key={colIndex}>{value?.toString() || '-'}</TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Stack>
          </Paper>
        )}
      </Stack>
    </Box>
  );
};

export default QueryInterface; 