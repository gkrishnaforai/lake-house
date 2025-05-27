import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  IconButton,
  Tooltip,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress
} from '@mui/material';
import {
  PlayArrow as RunIcon,
  Save as SaveIcon,
  History as HistoryIcon,
  Code as FormatIcon,
  Assessment as PlanIcon,
  Delete as DeleteIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon
} from '@mui/icons-material';
import { CatalogService, getUserId } from '../services/catalogService';
import { SqlResults } from './SqlResults';

interface SqlEditorProps {
  tableName?: string;
}

interface SavedQuery {
  id: string;
  name: string;
  query: string;
  isFavorite: boolean;
}

interface ExecutionPlan {
  plan: string;
}

export const SqlEditor: React.FC<SqlEditorProps> = ({ tableName }) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedQueries, setSavedQueries] = useState<SavedQuery[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showPlan, setShowPlan] = useState(false);
  const [executionPlan, setExecutionPlan] = useState<ExecutionPlan | null>(null);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [queryName, setQueryName] = useState('');
  const [queryResults, setQueryResults] = useState<any>(null);
  const catalogService = new CatalogService();

  useEffect(() => {
    loadSavedQueries();
  }, []);

  const loadSavedQueries = async () => {
    try {
      const queries = await catalogService.getSavedQueries();
      setSavedQueries(queries);
    } catch (error) {
      console.error('Error loading saved queries:', error);
    }
  };

  const handleRunQuery = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const userId = getUserId();
      const result = await catalogService.executeQuery(query, userId);
      setQueryResults(result);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to execute query');
    } finally {
      setLoading(false);
    }
  };

  const handleFormatQuery = () => {
    // TODO: Implement query formatting
    console.log('Format query not implemented yet');
  };

  const handleShowPlan = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const plan = await catalogService.getQueryPlan(query);
      setExecutionPlan(plan);
      setShowPlan(true);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to get execution plan');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveQuery = async () => {
    if (!query.trim()) return;
    const name = prompt('Enter a name for this query:');
    if (!name) return;
    try {
      await catalogService.saveQuery({
        name,
        query,
        isFavorite: false
      });
      await loadSavedQueries();
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to save query');
    }
  };

  const handleToggleFavorite = async (queryId: string) => {
    try {
      await catalogService.toggleQueryFavorite(queryId);
      await loadSavedQueries();
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to toggle favorite');
    }
  };

  const handleDeleteQuery = async (queryId: string) => {
    try {
      await catalogService.deleteQuery(queryId);
      await loadSavedQueries();
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to delete query');
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">SQL Editor</Typography>
            <Box>
              <Tooltip title="Show History">
                <IconButton onClick={() => setShowHistory(true)}>
                  <HistoryIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Show Execution Plan">
                <IconButton onClick={handleShowPlan} disabled={!query.trim()}>
                  <PlanIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <TextField
            fullWidth
            multiline
            rows={6}
            variant="outlined"
            placeholder="Write your SQL query here..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            sx={{ mb: 2 }}
          />

          <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
            <Button
              variant="contained"
              startIcon={<RunIcon />}
              onClick={handleRunQuery}
              disabled={loading || !query.trim()}
            >
              Run Query
            </Button>
            <Button
              variant="outlined"
              startIcon={<FormatIcon />}
              onClick={handleFormatQuery}
              disabled={!query.trim()}
            >
              Format
            </Button>
            <Button
              variant="outlined"
              startIcon={<SaveIcon />}
              onClick={() => setSaveDialogOpen(true)}
              disabled={!query.trim()}
            >
              Save
            </Button>
          </Box>

          {loading && (
            <Box display="flex" justifyContent="center" p={2}>
              <CircularProgress />
            </Box>
          )}
        </CardContent>
      </Card>

      {queryResults && (
        <SqlResults
          results={queryResults}
          loading={loading}
          error={error}
        />
      )}

      {/* Saved Queries Dialog */}
      <Dialog
        open={showHistory}
        onClose={() => setShowHistory(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Saved Queries</DialogTitle>
        <DialogContent>
          <List>
            {savedQueries.map((savedQuery) => (
              <React.Fragment key={savedQuery.id}>
                <ListItem>
                  <ListItemIcon>
                    <IconButton
                      onClick={() => handleToggleFavorite(savedQuery.id)}
                      size="small"
                    >
                      {savedQuery.isFavorite ? <StarIcon color="primary" /> : <StarBorderIcon />}
                    </IconButton>
                  </ListItemIcon>
                  <ListItemText
                    primary={savedQuery.name}
                  />
                  <Box>
                    <Tooltip title="Load Query">
                      <IconButton
                        size="small"
                        onClick={() => {
                          setQuery(savedQuery.query);
                          setShowHistory(false);
                        }}
                      >
                        <RunIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete Query">
                      <IconButton
                        size="small"
                        onClick={() => handleDeleteQuery(savedQuery.id)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </ListItem>
                <Divider />
              </React.Fragment>
            ))}
          </List>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowHistory(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Execution Plan Dialog */}
      <Dialog
        open={showPlan}
        onClose={() => setShowPlan(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Execution Plan</DialogTitle>
        <DialogContent>
          {executionPlan && (
            <Paper sx={{ p: 2, fontFamily: 'monospace' }}>
              <pre>{executionPlan.plan}</pre>
            </Paper>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowPlan(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Save Query Dialog */}
      <Dialog
        open={saveDialogOpen}
        onClose={() => setSaveDialogOpen(false)}
      >
        <DialogTitle>Save Query</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Query Name"
            value={queryName}
            onChange={(e) => setQueryName(e.target.value)}
            sx={{ mt: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={async () => {
              if (queryName.trim()) {
                await handleSaveQuery();
                setSaveDialogOpen(false);
              }
            }}
            variant="contained"
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SqlEditor; 