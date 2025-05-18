import React from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Tooltip,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Help as HelpIcon,
} from '@mui/icons-material';

interface AgentState {
  id: string;
  name: string;
  status: 'active' | 'inactive' | 'error' | 'unknown';
  lastSeen: string;
  currentTask?: string;
  error?: string;
}

interface AgentStatusPanelProps {
  agentStates: AgentState[];
}

const AgentStatusPanel: React.FC<AgentStatusPanelProps> = ({ agentStates }) => {
  const getStatusIcon = (status: AgentState['status']) => {
    switch (status) {
      case 'active':
        return <CheckCircleIcon color="success" />;
      case 'inactive':
        return <WarningIcon color="warning" />;
      case 'error':
        return <ErrorIcon color="error" />;
      default:
        return <HelpIcon color="disabled" />;
    }
  };

  const getStatusColor = (status: AgentState['status']): 'success' | 'warning' | 'error' | 'default' => {
    switch (status) {
      case 'active':
        return 'success';
      case 'inactive':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const formatLastSeen = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (diffInSeconds < 60) {
      return 'just now';
    } else if (diffInSeconds < 3600) {
      const minutes = Math.floor(diffInSeconds / 60);
      return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    } else if (diffInSeconds < 86400) {
      const hours = Math.floor(diffInSeconds / 3600);
      return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Agent Status
      </Typography>
      <List>
        {agentStates.map((agent) => (
          <ListItem key={agent.id}>
            <ListItemIcon>{getStatusIcon(agent.status)}</ListItemIcon>
            <ListItemText
              primary={agent.name}
              secondary={
                <Box>
                  {agent.currentTask && (
                    <Typography variant="body2" color="text.secondary">
                      Current Task: {agent.currentTask}
                    </Typography>
                  )}
                  {agent.error && (
                    <Typography variant="body2" color="error">
                      Error: {agent.error}
                    </Typography>
                  )}
                  <Typography variant="body2" color="text.secondary">
                    Last seen: {formatLastSeen(agent.lastSeen)}
                  </Typography>
                </Box>
              }
            />
            <Tooltip title={`Status: ${agent.status}`}>
              <Chip
                label={agent.status}
                color={getStatusColor(agent.status)}
                size="small"
              />
            </Tooltip>
          </ListItem>
        ))}
      </List>
    </Box>
  );
};

export default AgentStatusPanel; 