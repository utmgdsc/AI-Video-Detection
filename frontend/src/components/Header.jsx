export default function Header({ isHealthy }) {
  return (
    <header className="glass-panel">
      <div className="logo-container">
        <span className="title">AI Video Detection</span>
      </div>
      <div className="status-indicator">
        <div
          className={`status-dot ${
            isHealthy === null
              ? 'connecting'
              : isHealthy
              ? 'online'
              : 'offline'
          }`}
        ></div>
        <span>{isHealthy === null ? 'Checking Backend...' : isHealthy ? 'System Online' : 'Backend Offline'}</span>
      </div>
    </header>
  );
}
