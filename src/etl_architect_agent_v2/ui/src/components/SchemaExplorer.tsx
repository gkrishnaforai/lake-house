import React, { useState, useEffect } from 'react';
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
  CircularProgress,
  Alert
} from '@mui/material';
import { CatalogService } from '../services/catalogService';
import { TableInfo } from '../types/api';

interface SchemaExplorerProps {
  tableName: string;
  userId: string;
}

const SchemaExplorer: React.FC<SchemaExplorerProps> = ({ tableName, userId }) => {
  const [schema, setSchema] = useState<TableInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const catalogService = new CatalogService();

  useEffect(() => {
    const fetchSchema = async () => {
      if (!tableName) return;

      setLoading(true);
      setError(null);

      try {
        const tables = await catalogService.listTables(userId);
        const tableInfo = tables.find(t => t.name === tableName);
        if (tableInfo) {
          setSchema(tableInfo);
        } else {
          throw new Error(`Table ${tableName} not found`);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch schema');
      } finally {
        setLoading(false);
      }
    };

    fetchSchema();
  }, [tableName, userId]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 3 }}>
        {error}
      </Alert>
    );
  }

  if (!schema || !schema.columns || schema.columns.length === 0) {
    return (
      <Alert severity="info">
        No schema information available for this table
      </Alert>
    );
  }

  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Column Name</TableCell>
            <TableCell>Data Type</TableCell>
            <TableCell>Description</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {schema.columns.map((column) => (
            <TableRow key={column.name}>
              <TableCell>{column.name}</TableCell>
              <TableCell>{column.type}</TableCell>
              <TableCell>{column.description || '-'}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default SchemaExplorer; 