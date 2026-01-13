# AgentBeats SWE Verified - Docker Image

FROM python:3.13-slim

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create the Ollama model during build
RUN ollama serve & \
    sleep 5 && \
    ollama create qwen2.5-coder:7b -f /app/qwen2_5_coder_7b.ollamaModelFile
# Create directory for data
RUN mkdir -p /app/data

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Expose ports for green agent and Ollama
EXPOSE 9020

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:9020/.well-known/agent.json')" || exit 1

# Default command: Run entrypoint (starts Ollama + green agent)
CMD ["/app/entrypoint.sh"]
