import { useState, useRef } from 'react';
import './FileUpload.css';

export default function FileUpload({ onUpload, isAnalyzing }) {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    inputRef.current.click();
  };

  const handleFile = (file) => {
    if (isAnalyzing) return;
    onUpload(file);
  };

  return (
    <div className="upload-container glass-panel">
      <div className="upload-content text-center">
        <h2 className="upload-title">Scan Video for Manipulation</h2>
        <p className="upload-subtitle">
          Upload a video file to run deeply integrated AI detection for face-swapping, lip-syncing, and synthetic generation.
        </p>
      </div>

      <div
        className={`drag-zone ${dragActive ? 'drag-active' : ''} ${isAnalyzing ? 'analyzing' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          className="file-input"
          accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,audio/wav"
          onChange={handleChange}
          disabled={isAnalyzing}
        />
        
        {isAnalyzing ? (
          <div className="analyzing-state">
            <div className="spinner"></div>
            <p>Analyzing media... This might take a few moments.</p>
          </div>
        ) : (
          <div className="upload-prompt">
            <p className="prompt-text">Drag and drop your video file here</p>
            <p className="prompt-subtext">or</p>
            <button className="btn-primary" onClick={onButtonClick}>Select a File</button>
            <p className="supported-formats">Supported: MP4, MOV, AVI, WEBM, WAV</p>
          </div>
        )}
      </div>
    </div>
  );
}
