# AgentBeats SWE Verified

An AI-powered multi-agent system for Software Engineering evaluation tasks. This project implements a collaborative agent framework using the A2A (Agent-to-Agent) protocol to automate repository setup, dependency management, and code analysis.

## 🚀 Features

- **Multi-Agent Architecture**: Green agent orchestrates evaluation while participant agents provide LLM-powered analysis
- **Automated Environment Setup**: Intelligent Python version detection and virtual environment creation
- **Dependency Management**: Automated parsing and installation of project dependencies
- **A2A Protocol Support**: Built on the A2A SDK for standardized agent communication
- **Parallel Processing**: Configurable concurrent row processing for dataset evaluation

## 📋 Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended for fast package management)
- Docker (optional, for containerized deployment)

## 🛠️ Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/CoGian/agentbeats-swe-verified.git
cd agentbeats-swe-verified

# Install dependencies using uv
uv sync

# Or create a virtual environment first
uv venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

uv pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/CoGian/agentbeats-swe-verified.git
cd agentbeats-swe-verified

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

# Install the package
pip install -e .
```

## ⚙️ Configuration

Create a `.env` file in the project root with your API keys and configuration:

```env
# LLM Provider Configuration
OPENAI_API_KEY=your_openai_api_key
# Or use other providers supported by LiteLLM
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_gemini_api_key

# Optional: Custom LLM settings
LLM_MODEL=gpt-4
```

## 🦙 Ollama Setup

This project uses Ollama for local LLM inference. Follow these steps to set up Ollama:

### Install Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com/download
```

### Create the Custom Model

```bash
# Create the qwen2.5-coder:7b model with custom configuration
ollama create qwen2.5-coder:7b -f qwen2_5_coder_7b.ollamaModelFile
```

### Start Ollama Server

```bash
# Start the Ollama server (runs on port 11434 by default)
ollama serve

# Verify the model is running
ollama ps
```

## 🏃 Running the Application

### Running a Scenario

The main entry point is `agentbeats-run` which orchestrates the multi-agent evaluation:

```bash
# Run the default SWE scenario
uv run agentbeats-run swe_scenario/scenario.toml

# Run with visible logs
uv run agentbeats-run swe_scenario/scenario.toml --show-logs

# Start agents only (for debugging)
uv run agentbeats-run swe_scenario/scenario.toml --serve-only
```

### Running Individual Agents

You can also run agents individually:

```bash
# Start the Green Agent (orchestrator)
uv run src/swe_green_agent/agent.py

# Start a participant agent (dummy LLM for testing)
uv run swe_scenario/dummy_llm.py --host 127.0.0.1 --port 9021
```

## 🐳 Docker

### Building the Docker Image

```bash
docker build -t agentbeats-swe-verified .
```

### Running with Docker

```bash
# Run with environment variables from .env file
docker run --gpus all --env-file .env -p 9020:9020 agentbeats-swe-verified
```

## 📁 Project Structure

```
agentbeats-swe-verified/
├── src/
│   ├── agentbeats/           # Core package
│   │   ├── client.py         # A2A client implementation
│   │   ├── client_cli.py     # CLI client for evaluation
│   │   ├── run_scenario.py   # Scenario runner
│   │   ├── repo_tools.py     # Repository utility tools
│   │   └── ...
│   └── swe_green_agent/      # Green agent implementation
│       └── agent.py          # Main orchestrator agent
├── swe_scenario/             # Scenario definitions
│   ├── scenario.toml         # Default scenario configuration
│   └── dummy_llm.py          # Test participant agent
├── data/                     # Evaluation datasets
├── pyproject.toml            # Project configuration
├── Dockerfile                # Container definition
└── README.md                 # This file
```

## 🔧 Scenario Configuration

Scenarios are defined in TOML format. Example (`swe_scenario/scenario.toml`):

```toml
[green_agent]
endpoint = "http://127.0.0.1:9020"
cmd = "python src/swe_green_agent/agent.py --host 127.0.0.1 --port 9020"

[[participants]]
role = "agent"
endpoint = "http://127.0.0.1:9021"
cmd = "python swe_scenario/dummy_llm.py --host 127.0.0.1 --port 9021"

[config]
max_concurrent_rows = 5  # Parallel processing configuration
```

## 📊 Data Format

The evaluation uses Parquet files with the following expected columns:
- `repo`: Repository name
- `base_commit`: Git commit hash for code analysis
- `environment_setup_commit`: Commit for environment setup
- `problem_statement`: Description of the task
- `hints_text`: Additional hints for the LLM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [A2A Protocol](https://github.com/google/a2a) - Agent-to-Agent communication protocol
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM interface
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager