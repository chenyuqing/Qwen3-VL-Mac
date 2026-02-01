#!/bin/bash
# stop.sh - Stop the Qwen3-VL Server

# Ensure we are in the script's directory
cd "$(dirname "$0")"

STOPPED=false

# Method 1: Check PID file
if [ -f server.pid ]; then
    PID=$(cat server.pid)
    echo "Found server.pid (PID: $PID). Stopping..."
    if kill $PID 2>/dev/null; then
        rm server.pid
        echo "Server stopped."
        STOPPED=true
    else
        echo "Could not kill PID $PID (maybe already stopped)."
        rm server.pid
    fi
fi

# Method 2: Fallback to pkill if not stopped
if [ "$STOPPED" = false ]; then
    echo "Checking for any running uvicorn processes..."
    if pgrep -f "uvicorn server:app" > /dev/null; then
        echo "Found running processes. Killing..."
        pkill -f "uvicorn server:app"
        echo "Server processes killed."
    else
        echo "No running server found."
    fi
fi
