import './AnalysisResult.css';

const MODEL_META = {
  efficientnet: {
    label: 'EfficientNet',
    desc: 'Video frame analysis',
  },
  mesonet: {
    label: 'MesoNet',
    desc: 'Facial mesoscopic features',
  },
  xceptionnet: {
    label: 'XceptionNet',
    desc: 'Deep feature extraction',
  },
  aasist: {
    label: 'AASIST',
    desc: 'Audio authenticity',
  },
};

export default function AnalysisResult({ result, onReset }) {
  if (!result) return null;

  const { is_real, confidence, audio_score, video_score, details, individual_prediction = {}, individual_scores = {} } = result;

  const getScoreColor = (score) => {
    if (score === null || score === undefined) return 'neutral';
    if (score > 0.7) return 'success';
    if (score < 0.3) return 'danger';
    return 'warning';
  };

  const getConfidenceLevel = (conf) => {
    if (conf > 0.8) return 'High Confidence';
    if (conf > 0.5) return 'Medium Confidence';
    return 'Low Confidence';
  };

  const modelResults = [
    { key: 'efficientnet', score: individual_scores.efficientnet_score },
    { key: 'mesonet',      score: individual_scores.mesonet_score },
    { key: 'xceptionnet',  score: individual_scores.xceptionnet_score },
    { key: 'aasist',       score: audio_score },
  ];

  return (
    <div className="result-container glass-panel">
      {/* Header */}
      <div className="result-header">
        <h2 className="result-title">Analysis Complete</h2>
        <div className={`verdict-badge ${is_real === null ? 'verdict-null' : is_real ? 'verdict-real' : 'verdict-fake'}`}>
          {is_real === null ? 'INCONCLUSIVE' : is_real ? 'AUTHENTIC' : 'MANIPULATED / DEEPFAKE'}
        </div>
      </div>

      {/* Ensemble Confidence */}
      <div className="confidence-section">
        <div className="confidence-label">
          <span>Ensemble Score</span>
          <div className="ensemble-score-row">
            <span className="ensemble-score-value">{(confidence * 100).toFixed(2)}%</span>
            <span className="ensemble-score-badge">{getConfidenceLevel(confidence)}</span>
          </div>
        </div>
        <div className="progress-bar-bg">
          <div
            className="progress-bar-fill"
            style={{ width: `${Math.min(100, Math.max(0, confidence * 100))}%` }}
          ></div>
        </div>
        <div className="confidence-sublabel">Fake probability (ensemble weighted average)</div>
      </div>

      {/* Audio + Video Combined Scores */}
      <div className="metrics-grid">
        <div className="metric-card glass-panel-inner">
          <span className="metric-label">Video Score</span>
          {video_score !== null && video_score !== undefined ? (
            <div className={`metric-value color-${getScoreColor(video_score)}`}>
              {(video_score * 100).toFixed(1)}%
            </div>
          ) : (
            <div className="metric-value color-neutral">N/A</div>
          )}
          <span className="metric-sublabel">Fake probability</span>
        </div>
        <div className="metric-card glass-panel-inner">
          <span className="metric-label">Audio Score</span>
          {audio_score !== null && audio_score !== undefined ? (
            <div className={`metric-value color-${getScoreColor(audio_score)}`}>
              {(audio_score * 100).toFixed(1)}%
            </div>
          ) : (
            <div className="metric-value color-neutral">N/A</div>
          )}
          <span className="metric-sublabel">Fake probability</span>
        </div>
      </div>

      {/* Individual Model Breakdown */}
      <div className="model-breakdown">
        <h3 className="breakdown-heading">Model Breakdown</h3>
        <div className="model-grid">
          {modelResults.map(({ key, score }) => {
            const meta = MODEL_META[key];
            const isNull = score === null || score === undefined;
            const isFake = score > 0.5;
            return (
              <div key={key} className="model-card glass-panel-inner">
                <div className="model-card-top">
                  <span className="model-name">{meta.label}</span>
                </div>
                <div className="model-desc">{meta.desc}</div>
                <div className={`model-verdict ${isNull ? 'model-null' : isFake ? 'model-fake' : 'model-real'}`}>
                  {isNull ? '—' : `${(score * 100).toFixed(1)}%`}
                </div>
                {!isNull && <div className="model-score-label">{isFake ? 'likely fake' : 'likely real'}</div>}
              </div>
            );
          })}
        </div>
      </div>

      {/* Details / Notes */}
      {details && (
        <div className="details-section">
          <h3 className="details-heading">Notes</h3>
          <p className="details-text">{details}</p>
        </div>
      )}

      <div className="action-row">
        <button className="btn-secondary" onClick={onReset}>
          Analyze Another File
        </button>
      </div>
    </div>
  );
}
