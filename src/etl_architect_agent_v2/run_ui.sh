#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Create virtual environment if it doesn't exist
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    python3 -m venv "$PROJECT_ROOT/venv"
    source "$PROJECT_ROOT/venv/bin/activate"
    pip install -r "$PROJECT_ROOT/requirements.txt"
else
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Install frontend dependencies
cd "$SCRIPT_DIR/ui"
if [ ! -d "node_modules" ]; then
    npm install
fi

# Start backend in background
cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/api/main.py" &
BACKEND_PID=$!

# Start frontend
cd "$SCRIPT_DIR/ui"
npm start

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT 