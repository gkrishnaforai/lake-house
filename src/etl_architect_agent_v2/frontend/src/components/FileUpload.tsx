import React, { useState } from 'react';
import { Box, Button, CircularProgress, Typography, Alert, Paper } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { styled } from '@mui/material/styles';
import axios from 'axios';

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

interface UploadProgress {
  status: string;
  progress: number;
  message: string;
  error?: string;
  details?: any;
}

interface QualityCheckConfig {
  enabled_metrics: string[];
  thresholds: Record<string, number>;
  schedule?: string;
}

interface FileUploadProps {
  onUploadComplete?: (response: any) => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onUploadComplete }) => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [qualityConfig, setQualityConfig] = useState<QualityCheckConfig>({
    enabled_metrics: ["completeness", "uniqueness", "consistency"],
    thresholds: {
      completeness: 0.95,
      uniqueness: 0.90,
      consistency: 0.85
    }
  });
  const [runQualityChecks, setRunQualityChecks] = useState(false);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    setProgress({
      status: 'starting',
      progress: 0,
      message: 'Starting upload...'
    });

    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', 'test_user');

    try {
      setProgress({
        status: 'uploading',
        progress: 25,
        message: 'Uploading file...'
      });

      const response = await axios.post(
        '/api/catalog/upload',
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          onUploadProgress: (progressEvent) => {
            const progress = progressEvent.total
              ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
              : 0;
            setProgress({
              status: 'uploading',
              progress: progress,
              message: 'Uploading file...'
            });
          },
        }
      );

      const data = response.data;

      if (!response.ok) {
        throw new Error(data.error || 'Upload failed');
      }

      if (data.status === 'error') {
        setError(data.error || 'Upload failed');
        setProgress({
          status: 'error',
          progress: 100,
          message: 'Upload failed',
          error: data.error,
          details: data.details
        });
      } else {
        setProgress({
          status: 'success',
          progress: 100,
          message: 'Upload completed successfully',
          details: data.details
        });
        onUploadComplete?.(data);

        // Run quality checks if enabled
        if (runQualityChecks) {
          await configureAndRunQualityChecks(data.table_name);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setProgress({
        status: 'error',
        progress: 100,
        message: 'Upload failed',
        error: err instanceof Error ? err.message : 'Upload failed'
      });
    } finally {
      setUploading(false);
    }
  };

  const configureAndRunQualityChecks = async (tableName: string) => {
    try {
      // Configure quality checks
      await axios.post(
        `/api/catalog/tables/${tableName}/quality/config`,
        qualityConfig,
        {
          params: { user_id: "test_user" }
        }
      );

      // Run quality checks
      const response = await axios.post(
        `/api/catalog/tables/${tableName}/quality/run`,
        null,
        {
          params: { 
            user_id: "test_user",
            force: true
          }
        }
      );

      // Update quality metrics
      setProgress({
        status: 'success',
        progress: 100,
        message: 'Quality checks completed successfully',
        details: response.data
      });
    } catch (err) {
      console.error("Error running quality checks:", err);
      setError("Failed to run quality checks");
      setProgress({
        status: 'error',
        progress: 100,
        message: 'Failed to run quality checks',
        error: err instanceof Error ? err.message : 'Failed to run quality checks'
      });
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3, maxWidth: 600, mx: 'auto', my: 2 }}>
      <Box sx={{ textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          Upload File
        </Typography>
        
        <Button
          component="label"
          variant="contained"
          startIcon={<CloudUploadIcon />}
          disabled={uploading}
          sx={{ my: 2 }}
        >
          Choose File
          <VisuallyHiddenInput type="file" onChange={handleFileUpload} />
        </Button>

        {uploading && (
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', my: 2 }}>
            <CircularProgress size={24} sx={{ mr: 2 }} />
            <Typography>
              {progress?.message || 'Uploading...'}
            </Typography>
          </Box>
        )}

        {progress && !uploading && (
          <Box sx={{ mt: 2 }}>
            <Typography
              color={progress.status === 'success' ? 'success.main' : 'error.main'}
              variant="body1"
            >
              {progress.message}
            </Typography>
            {progress.details && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {JSON.stringify(progress.details, null, 2)}
              </Typography>
            )}
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        <div className="mt-4">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={runQualityChecks}
              onChange={(e) => setRunQualityChecks(e.target.checked)}
              className="form-checkbox h-4 w-4 text-blue-600"
            />
            <span>Run quality checks after upload</span>
          </label>
        </div>

        {runQualityChecks && (
          <div className="mt-4 p-4 border rounded-lg">
            <h3 className="text-lg font-semibold mb-2">Quality Check Configuration</h3>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Enabled Metrics</label>
              <div className="space-y-2">
                {["completeness", "uniqueness", "consistency"].map((metric) => (
                  <label key={metric} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={qualityConfig.enabled_metrics.includes(metric)}
                      onChange={(e) => {
                        const newMetrics = e.target.checked
                          ? [...qualityConfig.enabled_metrics, metric]
                          : qualityConfig.enabled_metrics.filter((m) => m !== metric);
                        setQualityConfig({ ...qualityConfig, enabled_metrics: newMetrics });
                      }}
                      className="form-checkbox h-4 w-4 text-blue-600"
                    />
                    <span className="capitalize">{metric}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Thresholds</label>
              {Object.entries(qualityConfig.thresholds).map(([metric, value]) => (
                <div key={metric} className="flex items-center space-x-2 mb-2">
                  <span className="w-32 capitalize">{metric}</span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={value}
                    onChange={(e) =>
                      setQualityConfig({
                        ...qualityConfig,
                        thresholds: {
                          ...qualityConfig.thresholds,
                          [metric]: parseFloat(e.target.value),
                        },
                      })
                    }
                    className="flex-1"
                  />
                  <span className="w-12 text-right">{value}</span>
                </div>
              ))}
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Schedule (Cron Expression)</label>
              <input
                type="text"
                value={qualityConfig.schedule || ""}
                onChange={(e) =>
                  setQualityConfig({ ...qualityConfig, schedule: e.target.value })
                }
                placeholder="0 0 * * * (daily at midnight)"
                className="w-full p-2 border rounded"
              />
            </div>
          </div>
        )}
      </Box>
    </Paper>
  );
}; 