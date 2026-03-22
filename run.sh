#!/bin/bash

# Ensure we're in the project root
cd "$(dirname "$0")" || exit

echo "Starting AI Video Detection Application..."

# Function to clean up background processes on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    # Suppress output from kill commands
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "App stopped successfully."
    exit 0
}

# Trap Ctrl+C (SIGINT) and termination signals (SIGTERM)
trap cleanup SIGINT SIGTERM

# 1. Start backend server
echo "Starting FastAPI backend on http://127.0.0.1:8000..."
# Using environment variables to configure max upload sizes if needed
AIVD_MAX_UPLOAD_SIZE_BYTES=$((1024 * 1024 * 1024)) python3 -m uvicorn backend.api.app:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Give the backend a quick second to spool up
sleep 1

# 2. Start frontend server
echo "Starting Vite frontend..."
cd frontend || { echo "Frontend directory not found!"; exit 1; }
npm run dev &
FRONTEND_PID=$!

echo "========================================="
echo "Both servers are running!"
echo "Backend: http://127.0.0.1:8000"
echo "Frontend: Check Vite terminal output above (usually http://localhost:5173)"
echo "Press Ctrl+C to stop everything."
echo "========================================="

# Wait indefinitely until interrupted
wait
