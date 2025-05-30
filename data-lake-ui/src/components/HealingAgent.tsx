import React, { useEffect, useState, useContext } from 'react';
import { Box, Typography, Button, Paper, Collapse, CircularProgress, Stack, Alert } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import BugReportIcon from '@mui/icons-material/BugReport';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { ErrorContext } from '../App';

interface LogEntry {
  timestamp: string;
  event_type: string;
  workflow_id?: string;
  node_id?: string;
  node_type?: string;
  exception?: string;
  result?: any;
  [key: string]: any;
}

interface AWSCredentialError {
  status: 'expired' | 'error';
  message: string;
  suggestions?: string[];
  error_details: {
    code: string;
    message: string;
  };
}

const checkBackendHealth = async (): Promise<boolean> => {
  try {
    const resp = await fetch('/api/healing/health');
    if (!resp.ok) {
      console.warn('Backend health check failed with status:', resp.status);
      return false;
    }
    const contentType = resp.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      console.warn('Backend returned non-JSON response:', contentType);
      return false;
    }
    const data = await resp.json();
    return data.status === 'healthy';
  } catch (error) {
    console.warn('Backend health check failed:', error);
    return false;
  }
};

const fetchRecentLogs = async (): Promise<LogEntry[]> => {
  try {
    const isHealthy = await checkBackendHealth();
    if (!isHealthy) {
      return [{
        timestamp: new Date().toISOString(),
        event_type: 'workflow_exception',
        exception: 'Backend service is unavailable',
        details: 'Please check if the backend server is running'
      }];
    }

    const resp = await fetch('/api/healing/logs/recent');
    if (!resp.ok) {
      const errorText = await resp.text();
      let errorDetails;
      try {
        errorDetails = JSON.parse(errorText);
      } catch {
        errorDetails = { message: errorText };
      }
      throw new Error(`Failed to fetch logs: ${errorDetails.message || resp.statusText}`);
    }
    const contentType = resp.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      throw new Error('Backend returned non-JSON response');
    }
    const logs = await resp.json();
    return logs;
  } catch (error) {
    console.warn('Error fetching logs:', error);
    return [{
      timestamp: new Date().toISOString(),
      event_type: 'workflow_exception',
      exception: 'Failed to fetch logs',
      details: error instanceof Error ? error.message : 'Unknown error'
    }];
  }
};

const checkAWSCredentials = async (): Promise<AWSCredentialError | null> => {
  try {
    const resp = await fetch('/api/healing/aws-credentials/status');
    const data = await resp.json();
    if (data.status === 'expired' || data.status === 'error') {
      return data as AWSCredentialError;
    }
    return null;
  } catch (e) {
    console.error('Error checking AWS credentials:', e);
    return null;
  }
};

const retryOperation = async (log: LogEntry): Promise<void> => {
  const resp = await fetch('/api/healing/retry', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      workflow_id: log.workflow_id,
      node_id: log.node_id,
      event_type: log.event_type 
    })
  });
  if (!resp.ok) {
    const error = await resp.text();
    throw new Error(`Failed to retry operation: ${error}`);
  }
};

const reportToSupport = async (log: LogEntry): Promise<void> => {
  const resp = await fetch('/api/healing/support/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(log)
  });
  if (!resp.ok) {
    const error = await resp.text();
    throw new Error(`Failed to report to support: ${error}`);
  }
};

const autoFix = async (log: LogEntry): Promise<void> => {
  const resp = await fetch('/api/healing/auto-fix', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(log)
  });
  if (!resp.ok) {
    const error = await resp.text();
    throw new Error(`Failed to auto-fix: ${error}`);
  }
};

const getContextSensitiveSuggestions = (log: LogEntry): string[] => {
  const suggestions: string[] = [];
  
  if (log.event_type === 'node_failure') {
    if (log.node_type === 'data_validation') {
      suggestions.push('Check data quality rules and input data format');
      suggestions.push('Verify data schema matches expected format');
    } else if (log.node_type === 'transformation') {
      suggestions.push('Review transformation logic and input/output mappings');
      suggestions.push('Check for data type mismatches in transformation');
    } else if (log.node_type === 'load') {
      suggestions.push('Verify target system connectivity and permissions');
      suggestions.push('Check if target system has sufficient capacity');
    }
  } else if (log.event_type === 'workflow_failure') {
    suggestions.push('Review workflow dependencies and execution order');
    suggestions.push('Check resource allocation and system limits');
  } else if (log.event_type === 'workflow_exception') {
    suggestions.push('Verify system configuration and environment variables');
    suggestions.push('Check for network connectivity issues');
  }

  // Add common suggestions
  suggestions.push('Review error logs for detailed information');
  suggestions.push('Check system resources and performance metrics');
  
  return suggestions;
};

const summarizeError = (log: LogEntry) => {
  if (log.event_type === 'node_failure') {
    return `Step "${log.node_id}" failed in workflow "${log.workflow_id}". Reason: ${log.exception}`;
  }
  if (log.event_type === 'workflow_failure') {
    return `Workflow "${log.workflow_id}" failed at node "${log.failed_node}". Reason: ${log.exception}`;
  }
  if (log.event_type === 'workflow_exception') {
    return `Workflow "${log.workflow_id}" encountered an exception: ${log.exception}`;
  }
  return log.exception || 'An error occurred';
};

// Add a function to check for API errors
const checkForAPIErrors = async (): Promise<LogEntry | null> => {
  try {
    // Check catalog API for recent errors
    const resp = await fetch('/api/catalog/tables/recent');
    if (!resp.ok) {
      const errorText = await resp.text();
      let errorDetails;
      try {
        errorDetails = JSON.parse(errorText);
      } catch {
        errorDetails = { message: errorText };
      }
      return {
        timestamp: new Date().toISOString(),
        event_type: 'workflow_exception',
        exception: 'Catalog API Error',
        details: errorDetails.message || resp.statusText,
        source: 'catalog_api'
      };
    }
    return null;
  } catch (error) {
    console.warn('Error checking API status:', error);
    return null;
  }
};

// Add a function to handle API responses
const handleAPIResponse = async (response: Response): Promise<Response> => {
  if (!response.ok) {
    const errorText = await response.text();
    let errorDetails;
    try {
      errorDetails = JSON.parse(errorText);
    } catch {
      errorDetails = { message: errorText };
    }

    // Check for specific error types
    if (response.status === 500) {
      const errorLog: LogEntry = {
        timestamp: new Date().toISOString(),
        event_type: 'workflow_exception',
        exception: 'Backend Error',
        details: errorDetails.message || 'Internal Server Error',
        source: 'backend',
        status: response.status
      };
      throw errorLog;
    }

    // Check for NotImplementedError
    if (errorDetails.message?.includes('NotImplementedError')) {
      const errorLog: LogEntry = {
        timestamp: new Date().toISOString(),
        event_type: 'workflow_exception',
        exception: 'Feature Not Implemented',
        details: 'This feature is not yet implemented in the backend',
        source: 'backend',
        status: response.status
      };
      throw errorLog;
    }

    const errorLog: LogEntry = {
      timestamp: new Date().toISOString(),
      event_type: 'workflow_exception',
      exception: `API Error: ${response.status} ${response.statusText}`,
      details: errorDetails.message || 'Unknown error',
      source: 'api',
      status: response.status
    };
    throw errorLog;
  }
  return response;
};

const HealingAgent: React.FC = () => {
  const { error, setError } = useContext(ErrorContext);
  const [errorLog, setErrorLog] = useState<LogEntry | null>(error);
  const [awsCredentialError, setAWSCredentialError] = useState<AWSCredentialError | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [lastChecked, setLastChecked] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const MAX_RETRIES = 3;

  // Add global error handler
  useEffect(() => {
    const handleGlobalError = (event: ErrorEvent) => {
      console.warn('Global error caught:', event.error);
      const errorLog: LogEntry = {
        timestamp: new Date().toISOString(),
        event_type: 'workflow_exception',
        exception: event.error.message || 'An unexpected error occurred',
        details: event.error.stack || 'No stack trace available'
      };
      setErrorLog(errorLog);
      setError(errorLog);
    };

    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.warn('Unhandled promise rejection:', event.reason);
      if (event.reason instanceof Error) {
        const errorLog: LogEntry = {
          timestamp: new Date().toISOString(),
          event_type: 'workflow_exception',
          exception: event.reason.message,
          details: event.reason.stack || 'No stack trace available'
        };
        setErrorLog(errorLog);
        setError(errorLog);
      }
    };

    // Add fetch error interceptor
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      try {
        const response = await originalFetch(...args);
        return handleAPIResponse(response);
      } catch (error) {
        if (error instanceof Error) {
          const errorLog: LogEntry = {
            timestamp: new Date().toISOString(),
            event_type: 'workflow_exception',
            exception: error.message,
            details: error.stack || 'No stack trace available',
            source: 'network'
          };
          setErrorLog(errorLog);
          setError(errorLog);
        } else if (typeof error === 'object' && error !== null && 'event_type' in error) {
          // This is our custom error log
          setErrorLog(error as LogEntry);
          setError(error as LogEntry);
        }
        throw error;
      }
    };

    window.addEventListener('error', handleGlobalError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      window.removeEventListener('error', handleGlobalError);
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
      window.fetch = originalFetch;
    };
  }, [setError]);

  // Update polling effect
  useEffect(() => {
    if (error) {
      setErrorLog(error);
      setRetryCount(0);
    } else {
      let interval: NodeJS.Timeout;
      const pollLogs = async () => {
        if (retryCount >= MAX_RETRIES) {
          console.log('Max retries reached, stopping polling');
          return;
        }

        setLoading(true);
        setApiError(null);
        try {
          // First check for API errors
          const apiError = await checkForAPIErrors();
          if (apiError) {
            setErrorLog(apiError);
            setError(apiError);
            setLastChecked(apiError.timestamp);
            setSuccess(false);
            setLoading(false);
            return;
          }

          const isHealthy = await checkBackendHealth();
          if (!isHealthy) {
            setRetryCount(prev => prev + 1);
            const errorLog: LogEntry = {
              timestamp: new Date().toISOString(),
              event_type: 'workflow_exception',
              exception: 'Backend service is unavailable',
              details: 'Please check if the backend server is running'
            };
            setErrorLog(errorLog);
            setError(errorLog);
            setSuccess(false);
            return;
          }

          // Reset retry count on successful health check
          setRetryCount(0);

          const logs = await fetchRecentLogs();
          const error = logs.reverse().find(
            (log) =>
              log.event_type === 'node_failure' ||
              log.event_type === 'workflow_failure' ||
              log.event_type === 'workflow_exception'
          );
          
          if (error && error.timestamp !== lastChecked) {
            setErrorLog(error);
            setError(error);
            setLastChecked(error.timestamp);
            setSuccess(false);
          } else if (!error) {
            setErrorLog(null);
            setError(null);
            setSuccess(true);
            setTimeout(() => {
              setSuccess(false);
            }, 3000);
          }
        } catch (e) {
          console.warn('Error polling logs:', e);
          setRetryCount(prev => prev + 1);
          const errorLog: LogEntry = {
            timestamp: new Date().toISOString(),
            event_type: 'workflow_exception',
            exception: e instanceof Error ? e.message : 'Failed to fetch logs',
            details: e instanceof Error ? e.stack : 'No stack trace available'
          };
          setErrorLog(errorLog);
          setError(errorLog);
          setSuccess(false);
        } finally {
          setLoading(false);
        }
      };
      pollLogs();
      interval = setInterval(pollLogs, 10000);
      return () => clearInterval(interval);
    }
  }, [error, lastChecked, retryCount, setError]);

  const handleRetry = async () => {
    if (!errorLog) return;
    setActionLoading('retry');
    setApiError(null);
    try {
      await retryOperation(errorLog);
      // Check if the operation was successful
      const isHealthy = await checkBackendHealth();
      if (isHealthy) {
        setSuccess(true);
        setErrorLog(null);
        // Auto-dismiss success message after 3 seconds
        setTimeout(() => {
          setSuccess(false);
        }, 3000);
      }
    } catch (e) {
      console.error('Error retrying operation:', e);
      setApiError(e instanceof Error ? e.message : 'Failed to retry operation');
      setSuccess(false);
    } finally {
      setActionLoading(null);
    }
  };

  const handleReportToSupport = async () => {
    if (!errorLog) return;
    setActionLoading('report');
    setApiError(null);
    try {
      await reportToSupport(errorLog);
      setSuccess(true);
      // Auto-dismiss success message after 3 seconds
      setTimeout(() => {
        setSuccess(false);
      }, 3000);
    } catch (e) {
      console.error('Error reporting to support:', e);
      setApiError(e instanceof Error ? e.message : 'Failed to report to support');
      setSuccess(false);
    } finally {
      setActionLoading(null);
    }
  };

  const handleAutoFix = async () => {
    if (!errorLog) return;
    setActionLoading('autofix');
    setApiError(null);
    try {
      await autoFix(errorLog);
      // Check if the fix was successful
      const isHealthy = await checkBackendHealth();
      if (isHealthy) {
        setSuccess(true);
        setErrorLog(null);
        // Auto-dismiss success message after 3 seconds
        setTimeout(() => {
          setSuccess(false);
        }, 3000);
      }
    } catch (e) {
      console.error('Error auto-fixing:', e);
      setApiError(e instanceof Error ? e.message : 'Failed to auto-fix');
      setSuccess(false);
    } finally {
      setActionLoading(null);
    }
  };

  const handleDismiss = () => {
    setErrorLog(null);
    setError(null);
    setAWSCredentialError(null);
    setApiError(null);
    setSuccess(false);
  };

  if (loading && !errorLog) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" p={2}>
        <CircularProgress />
      </Box>
    );
  }

  if (success) {
    return (
      <Box position="fixed" bottom={24} right={24} zIndex={1300}>
        <Paper elevation={6} sx={{ p: 2, minWidth: 340, maxWidth: 480 }}>
          <Stack direction="row" spacing={1} alignItems="center">
            <CheckCircleIcon color="success" />
            <Typography variant="h6" color="success.main">
              Operation Successful
            </Typography>
          </Stack>
          <Typography variant="body1" sx={{ mt: 1 }}>
            The issue has been resolved. The system is now working normally.
          </Typography>
        </Paper>
      </Box>
    );
  }

  if (!errorLog) return null;

  const summary = summarizeError(errorLog);
  const details = JSON.stringify(errorLog, null, 2);
  const suggestions = awsCredentialError?.suggestions || getContextSensitiveSuggestions(errorLog);

  return (
    <Box position="fixed" bottom={24} right={24} zIndex={1300}>
      <Paper elevation={6} sx={{ p: 2, minWidth: 340, maxWidth: 480 }}>
        <Typography variant="h6" color="error" gutterBottom>
          Healing Agent: Issue Detected
        </Typography>
        
        {apiError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {apiError}
          </Alert>
        )}
        
        <Typography variant="body1" gutterBottom>
          {summary}
        </Typography>
        
        <Box mt={2} mb={2}>
          <Typography variant="subtitle2" color="textSecondary" gutterBottom>
            Suggested Actions:
          </Typography>
          <Stack spacing={1}>
            {suggestions.map((suggestion, index) => (
              <Typography key={index} variant="body2" color="textSecondary">
                • {suggestion}
              </Typography>
            ))}
          </Stack>
        </Box>

        <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
          {awsCredentialError ? (
            <Button
              variant="contained"
              startIcon={<RefreshIcon />}
              onClick={() => window.location.reload()}
              disabled={!!actionLoading}
              size="small"
            >
              Refresh Credentials
            </Button>
          ) : (
            <>
              <Button
                variant="contained"
                startIcon={<RefreshIcon />}
                onClick={handleRetry}
                disabled={!!actionLoading}
                size="small"
              >
                Retry
              </Button>
              <Button
                variant="outlined"
                startIcon={<BugReportIcon />}
                onClick={handleReportToSupport}
                disabled={!!actionLoading}
                size="small"
              >
                Report to Support
              </Button>
              <Button
                variant="outlined"
                startIcon={<AutoFixHighIcon />}
                onClick={handleAutoFix}
                disabled={!!actionLoading}
                size="small"
              >
                Auto-Fix
              </Button>
            </>
          )}
        </Stack>

        <Stack direction="row" spacing={1}>
          <Button size="small" onClick={() => setDetailsOpen((open) => !open)}>
            {detailsOpen ? 'Hide Details' : 'Show Details'}
          </Button>
          <Button size="small" color="inherit" onClick={handleDismiss}>
            Dismiss
          </Button>
        </Stack>

        <Collapse in={detailsOpen}>
          <Box mt={1}>
            <pre style={{ fontSize: 12, background: '#f5f5f5', padding: 8, borderRadius: 4 }}>
              {details}
            </pre>
          </Box>
        </Collapse>
      </Paper>
    </Box>
  );
};

export default HealingAgent; 