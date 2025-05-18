import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Alert,
  Button,
  Collapse,
  IconButton,
} from '@mui/material';
import {
  Close as CloseIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';

interface ErrorReporterProps {
  error: string | null;
  workflowState: any;
  onRecovery?: (errorData: any) => Promise<void>;
}

const ErrorReporter: React.FC<ErrorReporterProps> = ({
  error,
  workflowState,
  onRecovery,
}) => {
  const [expanded, setExpanded] = useState(false);
  const [recoveryAttempts, setRecoveryAttempts] = useState(0);

  const handleRecovery = async () => {
    if (onRecovery) {
      try {
        await onRecovery({
          error,
          workflowState,
          attempt: recoveryAttempts + 1,
        });
        setRecoveryAttempts(prev => prev + 1);
      } catch (err) {
        console.error('Recovery failed:', err);
      }
    }
  };

  if (!error) {
    return null;
  }

  const getErrorSeverity = (errorMessage: string): 'error' | 'warning' | 'info' => {
    if (errorMessage.toLowerCase().includes('critical') || errorMessage.toLowerCase().includes('fatal')) {
      return 'error';
    }
    if (errorMessage.toLowerCase().includes('warning')) {
      return 'warning';
    }
    return 'info';
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Error Report
        </Typography>
        <IconButton
          size="small"
          onClick={() => setExpanded(!expanded)}
          aria-label="expand error details"
        >
          {expanded ? <CloseIcon /> : <RefreshIcon />}
        </IconButton>
      </Box>

      <Alert
        severity={getErrorSeverity(error)}
        sx={{ mb: 2 }}
        action={
          onRecovery && (
            <Button
              color="inherit"
              size="small"
              onClick={handleRecovery}
              disabled={recoveryAttempts >= 3}
            >
              {recoveryAttempts >= 3 ? 'Max Attempts Reached' : 'Recover'}
            </Button>
          )
        }
      >
        {error}
      </Alert>

      <Collapse in={expanded}>
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Error Context
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Workflow ID: {workflowState?.id || 'N/A'}
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Step: {workflowState?.currentStep || 'N/A'}
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Timestamp: {new Date().toISOString()}
          </Typography>
          {recoveryAttempts > 0 && (
            <Typography variant="body2" color="text.secondary">
              Recovery attempts: {recoveryAttempts}
            </Typography>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
};

export default ErrorReporter; 