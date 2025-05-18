import React from 'react';
import { Container, Typography, Box } from '@mui/material';
import { FileUpload } from './components/FileUpload';

function App() {
  const handleUploadComplete = (response: any) => {
    console.log('Upload completed:', response);
    // You can add additional logic here, such as refreshing the file list
    // or showing a success notification
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Data Lakehouse Builder
        </Typography>
        
        <FileUpload onUploadComplete={handleUploadComplete} />
        
        {/* Add other components here */}
      </Box>
    </Container>
  );
}

export default App; 