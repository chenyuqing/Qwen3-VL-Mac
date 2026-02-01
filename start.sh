#!/bin/bash
# start.sh - Start the Qwen3-VL Server in the background

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Check if server is already running
if [ -f server.pid ]; then
    PID=$(cat server.pid)
    if ps -p $PID > /dev/null; then
        echo "Server is already running with PID $PID."
        exit 1
    else
        echo "Found stale server.pid. Removing..."
        rm server.pid
    fi
fi

# Start the server using uv
echo "Starting Qwen3-VL Server..."
uv run uvicorn server:app --host 127.0.0.1 --port 8000 --reload > server.log 2>&1 &

# Save PID
PID=$!
echo $PID > server.pid
echo "Server started successfully! (PID: $PID)"
echo "Logs are being written to server.log"
echo "Access the Web UI at: http://127.0.0.1:8000"
