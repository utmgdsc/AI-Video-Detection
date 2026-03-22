import { useState, useEffect } from 'react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import AnalysisResult from './components/AnalysisResult';
import { checkHealth, analyzeVideo } from './api/client';
import './App.css';

function App() {
  const [isHealthy, setIsHealthy] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Check initial health
    checkHealth().then(status => setIsHealthy(status));
    
    // Periodically check health every 30s
    const interval = setInterval(() => {
      checkHealth().then(status => setIsHealthy(status));
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const handleUpload = async (file) => {
    setError(null);
    setIsAnalyzing(true);
    setResult(null);

    try {
      const analysisResult = await analyzeVideo(file);
      setResult(analysisResult);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred during analysis.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="App">
      <Header isHealthy={isHealthy} />
      
      <main>
        {!result && (
          <FileUpload 
            onUpload={handleUpload} 
            isAnalyzing={isAnalyzing} 
          />
        )}

        {error && (
          <div className="error-message glass-panel" style={{ padding: '1rem', color: 'var(--error-color)', border: '1px solid var(--error-color)', maxWidth: '800px', width: '100%', margin: '0 auto' }}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <AnalysisResult 
            result={result} 
            onReset={handleReset} 
          />
        )}
      </main>
    </div>
  );
}

export default App;
