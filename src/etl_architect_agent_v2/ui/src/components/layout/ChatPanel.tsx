import React, { useState, useRef, useEffect } from 'react';
import {
  Drawer,
  Box,
  Typography,
  TextField,
  IconButton,
  Paper,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Tooltip,
  Alert
} from '@mui/material';
import {
  Send as SendIcon,
  Clear as ClearIcon,
  Help as HelpIcon
} from '@mui/icons-material';
import { chatService, ChatMessage } from '../../services/chatService';

interface ChatPanelProps {
  open: boolean;
  onClose: () => void;
  width: number;
}

const exampleQueries = [
  "Show me the top 10 rows from the sales table",
  "What's the average revenue by region?",
  "Find all customers who made purchases above $1000",
  "Show me the data quality metrics for the customers table"
];

export const ChatPanel: React.FC<ChatPanelProps> = ({
  open,
  onClose,
  width
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      text: input,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const response = await chatService.sendMessage(input);
      
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        text: response.message,
        sender: 'assistant',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);

      // If the response includes data, add it as a separate message
      if (response.data) {
        const dataMessage: ChatMessage = {
          id: (Date.now() + 2).toString(),
          text: JSON.stringify(response.data, null, 2),
          sender: 'assistant',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, dataMessage]);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to process request';
      setError(errorMessage);
      
      const errorResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        text: `Error: ${errorMessage}`,
        sender: 'assistant',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorResponse]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  const handleExampleClick = (query: string) => {
    setInput(query);
  };

  return (
    <Drawer
      variant="persistent"
      anchor="right"
      open={open}
      onClose={onClose}
      sx={{
        width: width,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: width,
          boxSizing: 'border-box',
          backgroundColor: 'background.default',
          borderLeft: '1px solid',
          borderColor: 'divider'
        },
      }}
    >
      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        height: '100%',
        bgcolor: 'background.paper'
      }}>
        {/* Header */}
        <Box sx={{ 
          p: 2, 
          borderBottom: '1px solid',
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <Typography variant="h6">Chat Assistant</Typography>
          <Tooltip title="Close">
            <IconButton onClick={onClose} size="small">
              <ClearIcon />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Example Queries */}
        <Paper sx={{ m: 2, p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Try asking about your data:
          </Typography>
          <List dense>
            {exampleQueries.map((query, index) => (
              <ListItem
                key={index}
                button
                onClick={() => handleExampleClick(query)}
                sx={{ borderRadius: 1 }}
              >
                <ListItemText
                  primary={query}
                  primaryTypographyProps={{
                    variant: 'body2',
                    color: 'text.secondary'
                  }}
                />
              </ListItem>
            ))}
          </List>
        </Paper>

        {/* Error Alert */}
        {error && (
          <Alert severity="error" sx={{ mx: 2, mb: 2 }}>
            {error}
          </Alert>
        )}

        {/* Messages */}
        <Box sx={{ 
          flexGrow: 1, 
          overflow: 'auto',
          p: 2,
          display: 'flex',
          flexDirection: 'column',
          gap: 2
        }}>
          {messages.map((message) => (
            <Box
              key={message.id}
              sx={{
                display: 'flex',
                justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start',
              }}
            >
              <Paper
                sx={{
                  p: 2,
                  maxWidth: '80%',
                  backgroundColor: message.sender === 'user' ? 'primary.main' : 'background.default',
                  color: message.sender === 'user' ? 'primary.contrastText' : 'text.primary',
                  borderRadius: 2,
                  whiteSpace: 'pre-wrap',
                  fontFamily: message.sender === 'assistant' && message.text.startsWith('{') ? 'monospace' : 'inherit'
                }}
              >
                <Typography variant="body2">{message.text}</Typography>
                <Typography
                  variant="caption"
                  sx={{
                    display: 'block',
                    mt: 1,
                    color: message.sender === 'user' ? 'primary.contrastText' : 'text.secondary',
                    opacity: 0.7
                  }}
                >
                  {message.timestamp.toLocaleTimeString()}
                </Typography>
              </Paper>
            </Box>
          ))}
          <div ref={messagesEndRef} />
        </Box>

        {/* Input */}
        <Box sx={{ 
          p: 2, 
          borderTop: '1px solid',
          borderColor: 'divider',
          bgcolor: 'background.paper'
        }}>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about your data..."
            disabled={loading}
            InputProps={{
              endAdornment: (
                <IconButton
                  color="primary"
                  onClick={handleSend}
                  disabled={!input.trim() || loading}
                >
                  {loading ? <CircularProgress size={24} /> : <SendIcon />}
                </IconButton>
              ),
            }}
          />
        </Box>
      </Box>
    </Drawer>
  );
}; 