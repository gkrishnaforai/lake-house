import React from 'react';
import {
  Box,
  Typography,
  LinearProgress,
  CircularProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Pending as PendingIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';

interface WorkflowStep {
  id: string;
  name: string;
  status: 'completed' | 'in_progress' | 'pending' | 'error';
  progress: number;
  error?: string;
}

interface WorkflowState {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed';
  currentStep: string;
  steps: WorkflowStep[];
  startTime: string;
  endTime?: string;
}

interface ProgressTrackerProps {
  workflowState: WorkflowState | null;
}

const ProgressTracker: React.FC<ProgressTrackerProps> = ({ workflowState }) => {
  if (!workflowState) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="text.secondary">No active workflow</Typography>
      </Box>
    );
  }

  const getStepIcon = (status: WorkflowStep['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon color="success" />;
      case 'in_progress':
        return <PendingIcon color="primary" />;
      case 'error':
        return <ErrorIcon color="error" />;
      default:
        return <PendingIcon color="disabled" />;
    }
  };

  const getStepColor = (status: WorkflowStep['status']): 'success' | 'primary' | 'error' | 'secondary' => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'in_progress':
        return 'primary';
      case 'error':
        return 'error';
      default:
        return 'secondary';
    }
  };

  const calculateOverallProgress = () => {
    if (!workflowState.steps.length) return 0;
    const completedSteps = workflowState.steps.filter(
      step => step.status === 'completed'
    ).length;
    return (completedSteps / workflowState.steps.length) * 100;
  };

  return (
    <Box>
      <Box sx={{ mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Workflow Progress
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Box sx={{ flexGrow: 1, mr: 1 }}>
            <LinearProgress
              variant="determinate"
              value={calculateOverallProgress()}
              sx={{ height: 10, borderRadius: 5 }}
            />
          </Box>
          <Chip
            label={`${Math.round(calculateOverallProgress())}%`}
            color={workflowState.status === 'failed' ? 'error' : 'primary'}
          />
        </Box>
        <Typography variant="body2" color="text.secondary">
          {workflowState.name}
        </Typography>
      </Box>

      <List>
        {workflowState.steps.map((step) => (
          <ListItem key={step.id}>
            <ListItemIcon>{getStepIcon(step.status)}</ListItemIcon>
            <ListItemText
              primary={step.name}
              secondary={
                step.status === 'in_progress' ? (
                  <LinearProgress
                    variant="determinate"
                    value={step.progress}
                    color={getStepColor(step.status)}
                    sx={{ mt: 1 }}
                  />
                ) : step.error ? (
                  <Typography color="error" variant="body2">
                    {step.error}
                  </Typography>
                ) : null
              }
            />
            <Chip
              label={step.status.replace('_', ' ')}
              color={getStepColor(step.status)}
              size="small"
            />
          </ListItem>
        ))}
      </List>
    </Box>
  );
};

export default ProgressTracker; 