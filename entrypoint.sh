#!/bin/bash
set -e

# Start Ollama server in the background
echo "Starting Ollama server..."
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done
echo "Ollama is ready!"

# Run the green agent
echo "Starting green agent..."
exec uv run src/swe_green_agent/agent.py
