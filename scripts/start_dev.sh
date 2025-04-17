#!/bin/bash

# Function to kill processes on specified ports
kill_port() {
    local port=$1
    echo "Killing process on port $port..."
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
}

# Kill existing processes
kill_port 3000
kill_port 8000

# Start Next.js server
echo "Starting Next.js server..."
cd docs-site
npm run dev &
NEXT_PID=$!

# Start FastAPI server
echo "Starting FastAPI server..."
cd ..
python run_api.py &
API_PID=$!

# Function to handle cleanup
cleanup() {
    echo "Shutting down servers..."
    kill $NEXT_PID $API_PID
    exit 0
}

# Set up trap for SIGINT and SIGTERM
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait $NEXT_PID $API_PID 