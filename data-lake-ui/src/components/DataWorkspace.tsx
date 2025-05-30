import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Divider,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  IconButton,
  CircularProgress,
  Alert,
  Stack,
  Chip,
  Avatar
} from '@mui/material';
import Grid from '@mui/material/Grid';
import UploadIcon from '@mui/icons-material/Upload';
import AddIcon from '@mui/icons-material/Add';
import TableChartIcon from '@mui/icons-material/TableChart';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import { getTables, getTableFiles, uploadFile, getFilePreview } from '../api';

const EXAMPLE_TASKS = [
  'Create a new S3 bucket for raw data',
  'Ingest sales.xlsx as a new table called sales_data',
  'Show me the schema for customer_data',
  'Provision a Redshift cluster for analytics',
  'Run a data quality check on orders table',
  'Delete the table temp_uploads',
  'Import data from Google Sheets',
];

const DataWorkspace: React.FC = () => {
  // Catalog state
  const [tables, setTables] = useState<any[]>([]);
  const [selectedTable, setSelectedTable] = useState<any | null>(null);
  const [tableFiles, setTableFiles] = useState<any[]>([]);
  const [tablePreview, setTablePreview] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Upload dialog state
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [newTableName, setNewTableName] = useState('');

  // Chat state
  const [messages, setMessages] = useState<{ sender: 'user' | 'agent', text: string }[]>([
    { sender: 'agent', text: 'Hi! I am your Data Engineer Agent 🤖. How can I help you today?' }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  // Fetch tables on mount
  useEffect(() => {
    fetchTables();
  }, []);

  const fetchTables = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getTables();
      setTables(data);
      if (data.length > 0 && !selectedTable) setSelectedTable(data[0]);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch tables');
    } finally {
      setLoading(false);
    }
  };

  // Fetch files and preview for selected table
  useEffect(() => {
    if (selectedTable) {
      fetchTableFiles(selectedTable.name);
      fetchTablePreview(selectedTable.name);
    }
  }, [selectedTable]);

  const fetchTableFiles = async (tableName: string) => {
    try {
      const files = await getTableFiles(tableName);
      setTableFiles(files);
    } catch {
      setTableFiles([]);
    }
  };

  const fetchTablePreview = async (tableName: string) => {
    try {
      // Use the first file for preview if available
      const files = await getTableFiles(tableName);
      if (files.length > 0) {
        const preview = await getFilePreview(files[0].s3_path, 10);
        setTablePreview(preview);
      } else {
        setTablePreview([]);
      }
    } catch {
      setTablePreview([]);
    }
  };

  // Upload logic
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setNewTableName(file.name.split('.')[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || !newTableName) return;
    setLoading(true);
    setError(null);
    try {
      await uploadFile(newTableName, selectedFile);
      setUploadDialogOpen(false);
      setSelectedFile(null);
      setNewTableName('');
      await fetchTables();
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  // Chat logic
  const handleSend = async () => {
    if (!chatInput.trim()) return;
    setChatError(null);
    setMessages(msgs => [...msgs, { sender: 'user', text: chatInput }]);
    setChatInput('');
    setChatLoading(true);
    try {
      // Use your real chat API here
      const { sendChatMessage } = await import('../api');
      const res = await sendChatMessage({ message: chatInput });
      setMessages(msgs => [...msgs, { sender: 'agent', text: res.response || res.message }]);
    } catch (e: any) {
      setChatError(e.message || 'Failed to get agent response');
    } finally {
      setChatLoading(false);
    }
  };

  const handleExampleClick = (task: string) => {
    setChatInput(task);
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh', background: '#f7f9fb' }}>
      {/* Left: Table Explorer */}
      <Paper sx={{ width: 260, borderRight: '1px solid #e0e0e0', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h6"><TableChartIcon sx={{ mr: 1 }} />Tables</Typography>
          <IconButton onClick={() => setUploadDialogOpen(true)} size="small" color="primary">
            <AddIcon />
          </IconButton>
        </Box>
        <Divider />
        <List sx={{ flex: 1, overflow: 'auto' }}>
          {tables.map((table) => (
            <ListItem key={table.name} disablePadding>
              <ListItemButton
                selected={selectedTable?.name === table.name}
                onClick={() => setSelectedTable(table)}
              >
                <ListItemText
                  primary={table.name}
                  secondary={table.rowCount ? `${table.rowCount} rows` : ''}
                />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Paper>

      {/* Center: Table Details */}
      <Box sx={{ flex: 2, p: 3, overflow: 'auto' }}>
        {selectedTable ? (
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              <InsertDriveFileIcon sx={{ mr: 1, color: 'primary.main' }} />
              {selectedTable.name}
            </Typography>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              {selectedTable.description || 'No description'}
            </Typography>
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle1" gutterBottom>Schema</Typography>
            <Box sx={{ mb: 2 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', padding: 4 }}>Name</th>
                    <th style={{ textAlign: 'left', padding: 4 }}>Type</th>
                    <th style={{ textAlign: 'left', padding: 4 }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedTable.columns?.map((col: any) => (
                    <tr key={col.name}>
                      <td style={{ padding: 4 }}><b>{col.name}</b></td>
                      <td style={{ padding: 4 }}>{col.type}</td>
                      <td style={{ padding: 4 }}>{col.description || ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Box>
            <Typography variant="subtitle1" gutterBottom>Files</Typography>
            <List dense>
              {tableFiles.map(f => (
                <ListItem key={f.s3_path}>
                  <ListItemText
                    primary={f.file_name}
                    secondary={`${f.file_type}, ${Math.round(f.size / 1024)} KB`}
                  />
                </ListItem>
              ))}
            </List>
            <Typography variant="subtitle1" gutterBottom>Preview</Typography>
            <Box sx={{ overflowX: 'auto', mb: 2 }}>
              {tablePreview.length > 0 ? (
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      {Object.keys(tablePreview[0]).map(col => (
                        <th key={col} style={{ borderBottom: '1px solid #eee', padding: 4, textAlign: 'left' }}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {tablePreview.map((row, i) => (
                      <tr key={i}>
                        {Object.values(row).map((val, j) => (
                          <td key={j} style={{ padding: 4, borderBottom: '1px solid #f5f5f5' }}>{String(val)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <Typography variant="body2" color="text.secondary">No preview available.</Typography>
              )}
            </Box>
            <Button
              variant="outlined"
              startIcon={<UploadIcon />}
              onClick={() => setUploadDialogOpen(true)}
              sx={{ mt: 2 }}
            >
              Import Data
            </Button>
          </Paper>
        ) : (
          <Typography variant="body1" color="text.secondary">Select a table to view details.</Typography>
        )}
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <CircularProgress />
          </Box>
        )}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>
        )}
      </Box>

      {/* Right: Agent Chat */}
      <Paper sx={{ width: 380, borderLeft: '1px solid #e0e0e0', display: 'flex', flexDirection: 'column', p: 0 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', p: 2, borderBottom: '1px solid #eee', bgcolor: '#f5f7fa' }}>
          <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
            <SmartToyIcon />
          </Avatar>
          <Typography variant="h6" sx={{ flex: 1 }}>Data Engineer Agent</Typography>
        </Box>
        <Stack direction="row" spacing={1} sx={{ p: 2, flexWrap: 'wrap' }}>
          {EXAMPLE_TASKS.map(task => (
            <Chip key={task} label={task} onClick={() => handleExampleClick(task)} variant="outlined" size="small" sx={{ mb: 1 }} />
          ))}
        </Stack>
        <Divider />
        <Box sx={{ flex: 1, overflowY: 'auto', p: 2 }}>
          <List>
            {messages.map((msg, idx) => (
              <ListItem key={idx} sx={{ justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start' }}>
                <ListItemText
                  primary={msg.text}
                  sx={{
                    bgcolor: msg.sender === 'user' ? '#e3f2fd' : '#f1f8e9',
                    borderRadius: 2,
                    px: 2,
                    py: 1,
                    maxWidth: 300,
                    textAlign: msg.sender === 'user' ? 'right' : 'left',
                  }}
                />
              </ListItem>
            ))}
            {chatLoading && (
              <ListItem>
                <CircularProgress size={24} />
              </ListItem>
            )}
          </List>
          {chatError && <Alert severity="error" sx={{ mb: 1 }}>{chatError}</Alert>}
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 2, borderTop: '1px solid #eee' }}>
          <TextField
            fullWidth
            placeholder="Type your request..."
            value={chatInput}
            onChange={e => setChatInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') handleSend(); }}
          />
          <Button variant="contained" endIcon={<SmartToyIcon />} onClick={handleSend} disabled={!chatInput.trim() || chatLoading}>
            Send
          </Button>
        </Box>
      </Paper>

      {/* Upload Dialog */}
      <Dialog open={uploadDialogOpen} onClose={() => setUploadDialogOpen(false)}>
        <DialogTitle>Import Data</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <input
              type="file"
              style={{ display: 'none' }}
              id="file-upload-input"
              onChange={handleFileSelect}
            />
            <label htmlFor="file-upload-input">
              <Button
                variant="outlined"
                startIcon={<UploadIcon />}
                component="span"
                fullWidth
                sx={{ mb: 2 }}
              >
                Select File
              </Button>
            </label>
            {selectedFile && (
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Selected: {selectedFile.name}
              </Typography>
            )}
            <TextField
              fullWidth
              label="Table Name"
              value={newTableName}
              onChange={(e) => setNewTableName(e.target.value)}
              margin="normal"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleUpload}
            variant="contained"
            disabled={!selectedFile || !newTableName}
          >
            Import
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DataWorkspace; 