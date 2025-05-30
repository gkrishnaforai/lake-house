import React from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Tooltip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  TableChart as TableIcon,
  Assessment as QualityIcon,
  Storage as StorageIcon
} from '@mui/icons-material';

interface MetricCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
  trend?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, icon, color, trend }) => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Box sx={{ 
          backgroundColor: `${color}15`, 
          borderRadius: 1, 
          p: 1,
          mr: 2
        }}>
          {icon}
        </Box>
        <Typography variant="h6" component="div">
          {title}
        </Typography>
      </Box>
      <Typography variant="h4" component="div" sx={{ mb: 1 }}>
        {value}
      </Typography>
      {trend && (
        <Typography variant="body2" color="text.secondary">
          {trend}
        </Typography>
      )}
    </CardContent>
  </Card>
);

export const Dashboard: React.FC = () => {
  const [loading, setLoading] = React.useState(false);

  const handleRefresh = () => {
    setLoading(true);
    // TODO: Implement refresh logic
    setTimeout(() => setLoading(false), 1000);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Dashboard</Typography>
        <Tooltip title="Refresh Data">
          <IconButton onClick={handleRefresh} disabled={loading}>
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {loading && <LinearProgress sx={{ mb: 3 }} />}

      <Grid container spacing={3}>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Total Tables"
            value={42}
            icon={<TableIcon sx={{ color: 'primary.main' }} />}
            color="#1976d2"
            trend="+2 this week"
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Data Quality Score"
            value="92%"
            icon={<QualityIcon sx={{ color: 'success.main' }} />}
            color="#2e7d32"
            trend="+5% this month"
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Storage Used"
            value="1.2 TB"
            icon={<StorageIcon sx={{ color: 'warning.main' }} />}
            color="#ed6c02"
            trend="+150 GB this week"
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Active Users"
            value={156}
            icon={<CheckCircleIcon sx={{ color: 'info.main' }} />}
            color="#0288d1"
            trend="+12 this week"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Recent Data Quality Issues"
              action={
                <Tooltip title="View All">
                  <IconButton>
                    <WarningIcon color="warning" />
                  </IconButton>
                </Tooltip>
              }
            />
            <CardContent>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <ErrorIcon color="error" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Missing Values in Customer Table"
                    secondary="Last updated 2 hours ago"
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemIcon>
                    <WarningIcon color="warning" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Data Type Mismatch in Orders"
                    secondary="Last updated 5 hours ago"
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemIcon>
                    <ErrorIcon color="error" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Duplicate Records in Products"
                    secondary="Last updated 1 day ago"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Recent Table Updates"
              action={
                <Tooltip title="View All">
                  <IconButton>
                    <TableIcon color="primary" />
                  </IconButton>
                </Tooltip>
              }
            />
            <CardContent>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Sales Data Updated"
                    secondary="2 hours ago"
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Customer Profiles Refreshed"
                    secondary="5 hours ago"
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Inventory Data Synced"
                    secondary="1 day ago"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}; 