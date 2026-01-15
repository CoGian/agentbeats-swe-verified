# AgentBeats SWE Verified - Docker Image

FROM python:3.13-slim

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    patch \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN uv sync

# Create directory for data
RUN mkdir -p /app/data

# Default command: Run entrypoint (starts Ollama + green agent)
ENTRYPOINT ["uv", "run", "src/swe_green_agent/agent.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009