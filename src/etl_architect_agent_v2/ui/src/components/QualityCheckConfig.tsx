import React from 'react';
import {
  Box,
  Typography,
  FormControlLabel,
  Checkbox,
  Slider,
  TextField,
  Paper,
  Switch,
} from '@mui/material';

export interface QualityCheckConfig {
  enabled_metrics: string[];
  thresholds: Record<string, number>;
  schedule?: string;
}

interface QualityCheckConfigProps {
  config: QualityCheckConfig;
  onChange: (config: QualityCheckConfig) => void;
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
}

const DEFAULT_METRICS = ["completeness", "uniqueness", "consistency"];
const DEFAULT_THRESHOLDS = {
  completeness: 0.95,
  uniqueness: 0.90,
  consistency: 0.85
};

export const QualityCheckConfig: React.FC<QualityCheckConfigProps> = ({
  config,
  onChange,
  enabled,
  onEnabledChange,
}) => {
  const handleMetricToggle = (metric: string) => {
    const newMetrics = config.enabled_metrics.includes(metric)
      ? config.enabled_metrics.filter((m) => m !== metric)
      : [...config.enabled_metrics, metric];
    
    onChange({
      ...config,
      enabled_metrics: newMetrics,
    });
  };

  const handleThresholdChange = (metric: string, value: number) => {
    onChange({
      ...config,
      thresholds: {
        ...config.thresholds,
        [metric]: value,
      },
    });
  };

  const handleScheduleChange = (schedule: string) => {
    onChange({
      ...config,
      schedule,
    });
  };

  return (
    <Paper sx={{ p: 2, mt: 2 }}>
      <Box sx={{ mb: 2 }}>
        <FormControlLabel
          control={
            <Switch
              checked={enabled}
              onChange={(e) => onEnabledChange(e.target.checked)}
            />
          }
          label="Enable Quality Checks"
        />
      </Box>

      {enabled && (
        <>
          <Typography variant="subtitle1" gutterBottom>
            Enabled Metrics
          </Typography>
          <Box sx={{ mb: 2 }}>
            {DEFAULT_METRICS.map((metric) => (
              <FormControlLabel
                key={metric}
                control={
                  <Checkbox
                    checked={config.enabled_metrics.includes(metric)}
                    onChange={() => handleMetricToggle(metric)}
                  />
                }
                label={metric.charAt(0).toUpperCase() + metric.slice(1)}
              />
            ))}
          </Box>

          <Typography variant="subtitle1" gutterBottom>
            Thresholds
          </Typography>
          <Box sx={{ mb: 2 }}>
            {Object.entries(config.thresholds).map(([metric, value]) => (
              <Box key={metric} sx={{ mb: 2 }}>
                <Typography variant="body2" gutterBottom>
                  {metric.charAt(0).toUpperCase() + metric.slice(1)}
                </Typography>
                <Slider
                  value={value}
                  onChange={(_, newValue) => handleThresholdChange(metric, newValue as number)}
                  min={0}
                  max={1}
                  step={0.05}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                />
              </Box>
            ))}
          </Box>

          <Typography variant="subtitle1" gutterBottom>
            Schedule (Cron Expression)
          </Typography>
          <TextField
            fullWidth
            value={config.schedule || ""}
            onChange={(e) => handleScheduleChange(e.target.value)}
            placeholder="0 0 * * * (daily at midnight)"
            size="small"
            sx={{ mb: 2 }}
          />
        </>
      )}
    </Paper>
  );
}; 