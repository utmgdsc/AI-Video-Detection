// In production, set VITE_API_URL to the Modal endpoint URL.
// In dev, leave it unset — Vite's proxy handles /api and /health.
const API_BASE = import.meta.env.VITE_API_URL ?? '';

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health/ready`);
    if (!res.ok) return false;
    const data = await res.json();
    return data.ready;
  } catch (err) {
    console.error('Health check failed', err);
    return false;
  }
}

export async function analyzeVideo(file) {
  const formData = new FormData();
  formData.append('video_file', file);

  try {
    const res = await fetch(`${API_BASE}/api/v1/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => null);
      throw new Error(errorData?.detail || 'Analysis request failed');
    }

    const data = await res.json();
    return data.result;
  } catch (err) {
    console.error('Analysis failed', err);
    throw err;
  }
}
