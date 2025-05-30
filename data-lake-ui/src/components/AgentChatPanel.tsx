import React, { useState } from 'react';
import { Box, Paper, Typography, TextField, IconButton, Button, List, ListItem, ListItemText, InputAdornment, CircularProgress, Alert, Stack, Chip } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import { sendChatMessage } from '../api';

interface Message {
  sender: 'user' | 'agent';
  text: string;
}

const EXAMPLE_TASKS = [
  'Create a new S3 bucket for raw data',
  'Ingest sales.xlsx as a new table called sales_data',
  'Show me the schema for customer_data',
  'Provision a Redshift cluster for analytics',
  'Run a data quality check on orders table',
  'Delete the table temp_uploads',
  'Import data from Google Sheets',
];

const AgentChatPanel: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    { sender: 'agent', text: 'Hi! I am your Data Engineer Agent. How can I help you today?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;
    setError(null);
    const userMsg = { sender: 'user' as const, text: input };
    setMessages(msgs => [...msgs, userMsg]);
    setInput('');
    setLoading(true);
    try {
      const res = await sendChatMessage({ message: userMsg.text });
      setMessages(msgs => [...msgs, { sender: 'agent', text: res.response || res.message }]);
    } catch (e: any) {
      setError(e.message || 'Failed to get agent response');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setUploading(true);
      // TODO: Handle file upload in chat context
      setTimeout(() => {
        setMessages(msgs => [...msgs, { sender: 'agent', text: 'File received (stub).' }]);
        setUploading(false);
      }, 1000);
    }
  };

  const handleExampleClick = (task: string) => {
    setInput(task);
  };

  return (
    <Paper sx={{ maxWidth: 700, margin: '0 auto', p: 3, minHeight: 500, display: 'flex', flexDirection: 'column', height: '70vh' }}>
      <Typography variant="h6" gutterBottom>Agent Chat</Typography>
      <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: 'wrap' }}>
        {EXAMPLE_TASKS.map(task => (
          <Chip key={task} label={task} onClick={() => handleExampleClick(task)} variant="outlined" size="small" sx={{ mb: 1 }} />
        ))}
      </Stack>
      <List sx={{ flex: 1, overflowY: 'auto', mb: 2 }}>
        {messages.map((msg, idx) => (
          <ListItem key={idx} sx={{ justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start' }}>
            <ListItemText
              primary={msg.text}
              sx={{
                bgcolor: msg.sender === 'user' ? '#e3f2fd' : '#f1f8e9',
                borderRadius: 2,
                px: 2,
                py: 1,
                maxWidth: 400,
                textAlign: msg.sender === 'user' ? 'right' : 'left',
              }}
            />
          </ListItem>
        ))}
        {loading && (
          <ListItem>
            <CircularProgress size={24} />
          </ListItem>
        )}
      </List>
      {error && <Alert severity="error" sx={{ mb: 1 }}>{error}</Alert>}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <TextField
          fullWidth
          placeholder="Type your request..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') handleSend(); }}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton component="label" disabled={uploading}>
                  <AttachFileIcon />
                  <input type="file" hidden onChange={handleFileUpload} />
                </IconButton>
              </InputAdornment>
            )
          }}
        />
        <Button variant="contained" endIcon={<SendIcon />} onClick={handleSend} disabled={!input.trim() || loading}>
          Send
        </Button>
      </Box>
    </Paper>
  );
};

export default AgentChatPanel; 