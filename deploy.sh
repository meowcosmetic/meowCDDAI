#!/usr/bin/env bash
# AI-Enhanced Gaze Tracking System — Linux/macOS Deployment Script
# Usage: ./deploy.sh [--dev | --prod]

set -euo pipefail

MODE="${1:---dev}"

echo "============================================================"
echo " AI-Enhanced Gaze Tracking System Deployment"
echo " Mode: $MODE"
echo "============================================================"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10 or 3.12."
    exit 1
fi

# Activate virtual environment if present
if [ -f "venv312/bin/activate" ]; then
    echo "Activating virtual environment..."
    # shellcheck disable=SC1091
    source venv312/bin/activate
else
    echo "WARNING: No virtual environment found at venv312/"
    echo "Run: python3 -m venv venv312 && source venv312/bin/activate"
fi

# Install / update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Copy config template if no config exists
if [ ! -f "config.yaml" ]; then
    echo "Creating default config.yaml from template..."
    cp config_template.yaml config.yaml
fi

# Run tests before starting in production mode
if [ "$MODE" = "--prod" ]; then
    echo "Running test suite..."
    python -m pytest ai_enhanced_gaze_tracking/tests/ -q --tb=short
    echo "All tests passed."
fi

# Start the server
echo "Starting server..."
if [ "$MODE" = "--prod" ]; then
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
else
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload
fi
