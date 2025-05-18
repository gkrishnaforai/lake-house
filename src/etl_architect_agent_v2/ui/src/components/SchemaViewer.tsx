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
  Chip,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Grid,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  Info as InfoIcon,
  Warning as WarningIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon
} from '@mui/icons-material';
import { CatalogService } from '../services/catalogService';
import { TableInfo } from '../types/api';

interface SchemaViewerProps {
  selectedTables: TableInfo[];
}

interface ColumnInfo {
  name: string;
  type: string;
  nullable: boolean;
  description?: string;
  quality?: {
    completeness: number;
    uniqueness: number;
    validity: number;
  };
}

const SchemaViewer: React.FC<SchemaViewerProps> = ({ selectedTables }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [schemas, setSchemas] = useState<Record<string, ColumnInfo[]>>({});
  const catalogService = new CatalogService();

  useEffect(() => {
    const fetchSchemas = async () => {
      const newSchemas: Record<string, ColumnInfo[]> = {};
      for (const table of selectedTables) {
        try {
          const schema = await catalogService.getTableSchema(table.name);
          newSchemas[table.name] = schema.columns || [];
        } catch (error) {
          console.error(`Failed to fetch schema for table ${table.name}:`, error);
          newSchemas[table.name] = [];
        }
      }
      setSchemas(newSchemas);
    };

    if (selectedTables.length > 0) {
      fetchSchemas();
    } else {
      setSchemas({});
    }
  }, [selectedTables]);

  const getQualityColor = (score: number) => {
    if (score >= 90) return 'success';
    if (score >= 70) return 'warning';
    return 'error';
  };

  const getQualityIcon = (score: number) => {
    if (score >= 90) return <SuccessIcon fontSize="small" />;
    if (score >= 70) return <WarningIcon fontSize="small" />;
    return <ErrorIcon fontSize="small" />;
  };

  if (selectedTables.length === 0) {
    return (
      <Box>
        <Typography color="text.secondary">
          Select tables from the left panel to view their schemas.
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {selectedTables.map((table) => (
        <Box key={table.name} sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Schema for {table.name}
          </Typography>
          {schemas[table.name]?.length > 0 ? (
            <TableContainer component={Paper}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Column Name</TableCell>
                    <TableCell>Data Type</TableCell>
                    <TableCell>Nullable</TableCell>
                    <TableCell>Description</TableCell>
                    <TableCell>Quality Metrics</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {schemas[table.name].map((column, index) => (
                    <TableRow key={index}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {column.name}
                          {column.quality && (
                            <Tooltip title="Column Quality">
                              <IconButton size="small">
                                {getQualityIcon(
                                  (column.quality.completeness +
                                    column.quality.uniqueness +
                                    column.quality.validity) /
                                  3
                                )}
                              </IconButton>
                            </Tooltip>
                          )}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip label={column.type} size="small" />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={column.nullable ? 'Yes' : 'No'}
                          color={column.nullable ? 'warning' : 'success'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        {column.description || '-'}
                      </TableCell>
                      <TableCell>
                        {column.quality && (
                          <Box sx={{ display: 'flex', gap: 1 }}>
                            <Tooltip title={`Completeness: ${column.quality.completeness}%`}>
                              <Chip
                                icon={getQualityIcon(column.quality.completeness)}
                                label={`${column.quality.completeness}%`}
                                color={getQualityColor(column.quality.completeness)}
                                size="small"
                              />
                            </Tooltip>
                            <Tooltip title={`Uniqueness: ${column.quality.uniqueness}%`}>
                              <Chip
                                icon={getQualityIcon(column.quality.uniqueness)}
                                label={`${column.quality.uniqueness}%`}
                                color={getQualityColor(column.quality.uniqueness)}
                                size="small"
                              />
                            </Tooltip>
                            <Tooltip title={`Validity: ${column.quality.validity}%`}>
                              <Chip
                                icon={getQualityIcon(column.quality.validity)}
                                label={`${column.quality.validity}%`}
                                color={getQualityColor(column.quality.validity)}
                                size="small"
                              />
                            </Tooltip>
                          </Box>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Typography color="text.secondary">
              No schema information available for this table.
            </Typography>
          )}
        </Box>
      ))}
    </Box>
  );
};

export default SchemaViewer; 