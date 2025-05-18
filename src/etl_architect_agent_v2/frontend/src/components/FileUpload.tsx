import React, { useState } from 'react';
import { Box, Button, CircularProgress, Typography, Alert, Paper } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { styled } from '@mui/material/styles';

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

interface FileUploadProps {
  onUploadComplete?: (response: any) => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onUploadComplete }) => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

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

    try {
      setProgress({
        status: 'uploading',
        progress: 25,
        message: 'Uploading file...'
      });

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

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
      </Box>
    </Paper>
  );
}; 