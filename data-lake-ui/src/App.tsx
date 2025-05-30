import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import DataWorkspace from './components/DataWorkspace';
import './App.css';
import HealingAgent from './components/HealingAgent';

// Create a context for error handling
export const ErrorContext = React.createContext<{
  error: any | null;
  setError: (error: any | null) => void;
}>({
  error: null,
  setError: () => {},
});

const App: React.FC = () => {
  const [error, setError] = useState<any | null>(null);

  // Global error handler
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      console.error('Global error:', event.error);
      setError({
        event_type: 'workflow_exception',
        exception: event.error.message,
        timestamp: new Date().toISOString()
      });
    };

    // Handle fetch errors
    const handleFetchError = (event: PromiseRejectionEvent) => {
      console.error('Fetch error:', event.reason);
      setError({
        event_type: 'workflow_exception',
        exception: event.reason?.message || 'An error occurred',
        timestamp: new Date().toISOString()
      });
    };

    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleFetchError);
    
    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleFetchError);
    };
  }, []);

  return (
    <ErrorContext.Provider value={{ error, setError }}>
      <Router>
        <div className="app-container">
          <Routes>
            <Route path="/" element={<DataWorkspace />} />
          </Routes>
          {/* Always render HealingAgent, it will handle its own visibility */}
          <HealingAgent />
        </div>
      </Router>
    </ErrorContext.Provider>
  );
};

export default App;
