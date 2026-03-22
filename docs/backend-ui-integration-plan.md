# Backend Plan for UI Integration (Simplified)

This document defines a **local-first backend** for UI integration.
It intentionally avoids production-only complexity (queue workers, retries, distributed job orchestration).

## Goals

- Expose video analysis through a simple web API
- Support safe file uploads and temporary storage
- Load models once at startup (not per request)
- Keep the system easy to run/debug for project demos and development

## Scope

In scope:
- FastAPI API layer
- Synchronous inference endpoint
- Basic upload validation and cleanup
- Typed settings + path validation
- Health endpoints

Out of scope (for now):
- Async job queue (Celery/Redis)
- Job polling/cancellation endpoints
- Retry/timeouts orchestration
- Deployment hardening/scale concerns

## Simplified Architecture

1. Client uploads a video file to `POST /api/v1/analyze`
2. Backend saves upload to a temporary directory
3. Backend runs inference synchronously via shared `DetectorService`
4. Backend returns the result in the same response
5. Backend always cleans temporary files/directories in `finally`

## API Contract

Implemented endpoints:
- `GET /health/live`
- `GET /health/ready`
- `POST /api/v1/analyze` (multipart upload, returns immediate result)

No `/jobs/*` endpoints in this simplified version.

## Configuration Strategy

- Keep YAML config (`backend/config/ensemble.yaml`) as primary source
- Allow env overrides for local flexibility
- Validate critical values/paths at startup

Useful env vars:
- `AIVD_CONFIG_PATH`
- `AIVD_VALIDATE_PATHS`
- `AIVD_MAX_UPLOAD_SIZE_BYTES`
- `AIVD_ALLOWED_VIDEO_SUFFIXES`
- `AIVD_ALLOWED_VIDEO_MIME_TYPES`

## Upload & Temp File Handling

- Stream upload to disk in chunks (avoid loading full file in memory)
- Validate extension and MIME type
- Enforce max upload size
- Use isolated temp directory per request
- Cleanup temp artifacts immediately after request completes

## Model Lifecycle

- Build `DetectorService` at API startup
- Reuse loaded service for all requests
- Release resources on app shutdown

## Implementation Status

Completed:
- Step 1: Service extraction (`backend/services/detector_service.py`)
- Step 2: Typed settings + startup validation (`backend/core/config.py`)
- Step 3: FastAPI app + health + analyze route
- Step 4 (simplified): Upload validation + temp cleanup in request lifecycle

Intentionally removed to keep it simple:
- Job store and `/jobs/*` endpoints
- Background temp sweeper thread

## Suggested Next Steps (Simplified Track)

1. Run end-to-end smoke tests with sample videos.
2. Add 2-3 focused API tests for `/api/v1/analyze` success/error cases.
3. Improve error response consistency (small schema tweaks).
4. Wire a minimal frontend upload page to this endpoint.

## Runbook

```bash
pip install -r backend/requirements.txt
export AIVD_VALIDATE_PATHS=0
uvicorn backend.api.app:app --host 127.0.0.1 --port 8080 --reload
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8080/api/v1/analyze" \
  -F "video_file=@/absolute/path/to/video.mp4"
```
