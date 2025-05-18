import React from 'react';
import {
  Box,
  Typography,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Button,
  Chip,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Pending as PendingIcon,
} from '@mui/icons-material';

interface WorkflowStep {
  id: string;
  name: string;
  status: 'completed' | 'in_progress' | 'pending' | 'error';
  description?: string;
  error?: string;
  startTime?: string;
  endTime?: string;
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

interface WorkflowVisualizerProps {
  workflowState: WorkflowState | null;
  onStepClick?: (stepId: string) => void;
}

const WorkflowVisualizer: React.FC<WorkflowVisualizerProps> = ({
  workflowState,
  onStepClick,
}) => {
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

  const formatDuration = (startTime?: string, endTime?: string) => {
    if (!startTime) return '';
    const start = new Date(startTime);
    const end = endTime ? new Date(endTime) : new Date();
    const diffInSeconds = Math.floor((end.getTime() - start.getTime()) / 1000);

    if (diffInSeconds < 60) {
      return `${diffInSeconds}s`;
    } else if (diffInSeconds < 3600) {
      const minutes = Math.floor(diffInSeconds / 60);
      return `${minutes}m ${diffInSeconds % 60}s`;
    } else {
      const hours = Math.floor(diffInSeconds / 3600);
      const minutes = Math.floor((diffInSeconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Workflow Steps
      </Typography>
      <Stepper orientation="vertical">
        {workflowState.steps.map((step) => (
          <Step key={step.id} active={step.status === 'in_progress'} completed={step.status === 'completed'}>
            <StepLabel
              StepIconComponent={() => getStepIcon(step.status)}
              optional={
                <Chip
                  label={step.status.replace('_', ' ')}
                  color={getStepColor(step.status)}
                  size="small"
                />
              }
            >
              {step.name}
            </StepLabel>
            <StepContent>
              {step.description && (
                <Typography variant="body2" color="text.secondary" paragraph>
                  {step.description}
                </Typography>
              )}
              {step.error && (
                <Typography variant="body2" color="error" paragraph>
                  Error: {step.error}
                </Typography>
              )}
              {step.startTime && (
                <Typography variant="body2" color="text.secondary">
                  Duration: {formatDuration(step.startTime, step.endTime)}
                </Typography>
              )}
              {onStepClick && (
                <Box sx={{ mt: 1 }}>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => onStepClick(step.id)}
                  >
                    View Details
                  </Button>
                </Box>
              )}
            </StepContent>
          </Step>
        ))}
      </Stepper>
    </Box>
  );
};

export default WorkflowVisualizer; 