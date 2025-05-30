import React from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  IconButton,
  Tooltip,
  useTheme
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Assessment as QualityIcon,
  Schema as SchemaIcon,
  DataObject as DataIcon,
  ArrowForward as ArrowForwardIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const features = [
  {
    title: 'Upload Data',
    description: 'Upload and manage your data files in various formats',
    icon: <UploadIcon sx={{ fontSize: 40 }} />,
    path: '/upload'
  },
  {
    title: 'Data Quality',
    description: 'Analyze and monitor your data quality metrics',
    icon: <QualityIcon sx={{ fontSize: 40 }} />,
    path: '/quality'
  },
  {
    title: 'Schema Explorer',
    description: 'Explore and understand your data structure',
    icon: <SchemaIcon sx={{ fontSize: 40 }} />,
    path: '/schema'
  },
  {
    title: 'Data Explorer',
    description: 'Query and analyze your data interactively',
    icon: <DataIcon sx={{ fontSize: 40 }} />,
    path: '/explore'
  }
];

export const MainPage: React.FC = () => {
  const navigate = useNavigate();
  const theme = useTheme();

  return (
    <Box sx={{ p: 3 }}>
      {/* Welcome Section */}
      <Paper
        sx={{
          p: 4,
          mb: 4,
          background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.primary.dark} 90%)`,
          color: 'white',
          borderRadius: 2
        }}
      >
        <Typography variant="h4" gutterBottom>
          Welcome to Data Agent
        </Typography>
        <Typography variant="body1" sx={{ mb: 3, opacity: 0.9 }}>
          Your intelligent data management assistant. Upload, analyze, and explore your data with ease.
        </Typography>
        <Button
          variant="contained"
          color="secondary"
          size="large"
          onClick={() => navigate('/upload')}
          endIcon={<ArrowForwardIcon />}
        >
          Get Started
        </Button>
      </Paper>

      {/* Features Grid */}
      <Grid container spacing={3}>
        {features.map((feature) => (
          <Grid item xs={12} sm={6} md={3} key={feature.title}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: theme.shadows[4]
                }
              }}
            >
              <CardContent sx={{ flexGrow: 1 }}>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    mb: 2,
                    color: 'primary.main'
                  }}
                >
                  {feature.icon}
                  <Typography variant="h6" sx={{ ml: 1 }}>
                    {feature.title}
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
              <CardActions>
                <Button
                  size="small"
                  endIcon={<ArrowForwardIcon />}
                  onClick={() => navigate(feature.path)}
                >
                  Explore
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Quick Actions */}
      <Paper sx={{ mt: 4, p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Quick Actions
        </Typography>
        <Grid container spacing={2}>
          <Grid item>
            <Button
              variant="outlined"
              startIcon={<UploadIcon />}
              onClick={() => navigate('/upload')}
            >
              Upload New Data
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="outlined"
              startIcon={<QualityIcon />}
              onClick={() => navigate('/quality')}
            >
              Check Data Quality
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="outlined"
              startIcon={<SchemaIcon />}
              onClick={() => navigate('/schema')}
            >
              View Schema
            </Button>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
}; 